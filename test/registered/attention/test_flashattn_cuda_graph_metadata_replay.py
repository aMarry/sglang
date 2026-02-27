"""
Unit tests for FlashAttention CUDA graph metadata replay correctness.

These tests verify that the page_table and related buffers are properly
refreshed (and stale entries cleared) across multiple CUDA graph replays.
This is the root cause of the Mimo-v2 MTP second-round output error.

Usage:
    python3 -m pytest test_flashattn_cuda_graph_metadata_replay.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

# Unit tests for metadata update helpers (CPU runnable for CI)
# register_cuda_ci is not needed here since we run on CPU


class TestNormalDecodeSetMetadata(unittest.TestCase):
    """Tests for normal_decode_set_metadata helper function."""

    def setUp(self):
        from sglang.srt.layers.attention.flashattention_backend import (
            normal_decode_set_metadata,
        )

        self.fn = normal_decode_set_metadata
        self.page_size = 4
        self.bs = 2
        max_num_pages = 8
        device = torch.device("cpu")

        # Pre-allocated buffers (like init_cuda_graph_state creates)
        self.cache_seqlens = torch.zeros(self.bs, dtype=torch.int32, device=device)
        self.cu_seqlens_k = torch.zeros(self.bs + 1, dtype=torch.int32, device=device)
        self.page_table = torch.zeros(
            self.bs, max_num_pages, dtype=torch.int32, device=device
        )
        # req_to_token: each row maps seq positions to physical KV slots
        # With page_size=4, strided_indices = [0, 4, 8, ...]
        self.req_to_token = torch.zeros(
            self.bs, max_num_pages * self.page_size, dtype=torch.int32, device=device
        )
        for r in range(self.bs):
            for pos in range(max_num_pages * self.page_size):
                # page r starts at physical slot r * max_num_pages * page_size
                self.req_to_token[r, pos] = r * max_num_pages * self.page_size + pos

        self.req_pool_indices = torch.arange(self.bs, dtype=torch.int32, device=device)
        self.strided_indices = torch.arange(
            0, max_num_pages * self.page_size, self.page_size, device=device
        )

    def _call(self, seq_lens):
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
        max_len = max(seq_lens)
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size
        self.fn(
            self.cache_seqlens,
            self.cu_seqlens_k,
            self.page_table,
            self.req_to_token,
            self.req_pool_indices,
            self.strided_indices,
            max_seq_pages,
            seq_lens_t,
            0,
            self.page_size,
        )
        return max_seq_pages

    def test_page_table_unused_columns_zeroed_after_shrink(self):
        """
        When the batch's max sequence length decreases between replays (e.g., due
        to request replacement), the page_table columns beyond the new max must be
        zeroed to avoid stale data being accessed in subsequent replays.
        """
        # Replay 1: long sequences → 4 pages used
        max_pages_1 = self._call([16, 8])
        self.assertEqual(max_pages_1, 4)
        # Columns 0-3 should be non-zero (valid page indices)
        self.assertTrue((self.page_table[:, :max_pages_1] != 0).any())

        # Replay 2: shorter sequences (request replaced) → only 2 pages used
        # Manually set req_to_token to simulate a new shorter request replacing
        # the old longer one at row 0.
        new_req_to_token = torch.zeros_like(self.req_to_token)
        new_req_to_token[0, :4] = 100  # new short request, single page at slot 100
        new_req_to_token[1, :4] = self.req_to_token[1, :4]
        self.req_to_token.copy_(new_req_to_token)

        max_pages_2 = self._call([4, 4])
        self.assertEqual(max_pages_2, 1)

        # Columns 0 should have the new values
        expected_page_0_row0 = 100 // self.page_size  # = 25
        self.assertEqual(self.page_table[0, 0].item(), expected_page_0_row0)

        # Columns 1 and beyond MUST be zeroed (stale data cleared)
        self.assertTrue(
            (self.page_table[:, max_pages_2:] == 0).all(),
            f"Stale data found in page_table columns {max_pages_2}+: "
            f"{self.page_table[:, max_pages_2:]}",
        )

    def test_page_table_grows_correctly(self):
        """When sequence length grows, new page columns should be filled."""
        max_pages_1 = self._call([4, 4])  # 1 page each
        self.assertEqual(max_pages_1, 1)

        max_pages_2 = self._call([8, 8])  # 2 pages each
        self.assertEqual(max_pages_2, 2)

        # Both columns 0 and 1 should be set
        self.assertTrue((self.page_table[:, :max_pages_2] != 0).any())
        # Beyond max_pages_2 should still be 0 (never written)
        self.assertTrue((self.page_table[:, max_pages_2:] == 0).all())


class TestDraftDecodeSetExpandMetadata(unittest.TestCase):
    """Tests for draft_decode_set_expand_metadata helper function."""

    def setUp(self):
        from sglang.srt.layers.attention.flashattention_backend import (
            draft_decode_set_expand_metadata,
        )

        self.fn = draft_decode_set_expand_metadata
        self.bs = 2
        self.topk = 2
        self.page_size = 4
        self.decode_length = 2
        device = torch.device("cpu")
        num_seqs = self.bs * self.topk  # 4 virtual sequences

        # Pre-allocated expand metadata buffers
        self.cache_seqlens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
        self.cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        self.page_table = torch.zeros(
            num_seqs,
            self.decode_length + 1,  # decode_length + 1 extra for partial page
            dtype=torch.int32,
            device=device,
        )

    def _call(self, last_page_lens_list, cache_loc_list):
        last_page_lens = torch.tensor(
            last_page_lens_list, dtype=torch.int32, device="cpu"
        )
        cache_loc = torch.tensor(
            cache_loc_list, dtype=torch.int64, device="cpu"
        ).view(self.bs * self.topk, self.decode_length)
        self.fn(
            cache_seqlens_int32=self.cache_seqlens,
            cu_seqlens_k=self.cu_seqlens_k,
            page_table=self.page_table,
            last_page_lens=last_page_lens,
            decode_length=self.decode_length,
            cache_loc=cache_loc,
            topk=self.topk,
            page_size=self.page_size,
        )

    def test_cu_seqlens_k_updated_consistently_with_cache_seqlens(self):
        """
        When page_size > 1, last_page_lens can be non-zero.  cu_seqlens_k must
        match cumsum(cache_seqlens_int32) so the attention kernel sees correct
        per-sequence KV lengths.
        """
        # last_page_lens = [1, 2] for bs=2 requests
        # cache_loc: all draft tokens on page 5 (physical locations 20-23)
        last_page_lens = [1, 2]
        cache_loc = [
            [20, 21],
            [20, 21],
            [22, 23],
            [22, 23],
        ]  # [bs*topk, decode_length]
        self._call(last_page_lens, cache_loc)

        # Expected cache_seqlens: decode_length + expanded_last_page_lens
        # expanded = [1, 1, 2, 2] (topk=2, repeat_interleave)
        expected_cache_seqlens = torch.tensor(
            [
                self.decode_length + 1,
                self.decode_length + 1,
                self.decode_length + 2,
                self.decode_length + 2,
            ],
            dtype=torch.int32,
        )
        self.assertTrue(
            torch.equal(self.cache_seqlens, expected_cache_seqlens),
            f"cache_seqlens mismatch: {self.cache_seqlens} vs {expected_cache_seqlens}",
        )

        # cu_seqlens_k must be cumsum([0] + cache_seqlens)
        expected_cu_seqlens_k = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(expected_cache_seqlens, 0)]
        )
        self.assertTrue(
            torch.equal(self.cu_seqlens_k, expected_cu_seqlens_k),
            f"cu_seqlens_k mismatch: {self.cu_seqlens_k} vs {expected_cu_seqlens_k}",
        )

    def test_page_table_stale_data_cleared_between_replays(self):
        """
        When last_page_lens decreases between replays (e.g., seq_len crosses a
        page boundary), scatter_ only writes to fewer positions.  Without the
        pre-zero fix, stale page indices from the first replay remain in the
        table and are incorrectly accessed by the attention kernel.
        """
        # Replay 1: last_page_lens = 3, decode_length=2 → total 5 tokens → 2 pages
        # Physical pages: seq 0 uses pages [0,1], seq 1 uses pages [2,3]
        cache_loc_r1 = [
            [0, 4],   # seq 0 topk 0: page 0 (slots 0,1) and page 1 (slot 4)
            [0, 4],   # seq 0 topk 1: same
            [8, 12],  # seq 1 topk 0: pages 2,3
            [8, 12],  # seq 1 topk 1: same
        ]
        self._call([3, 3], cache_loc_r1)

        # After replay 1: page_table should have 2 filled columns
        pt_after_r1 = self.page_table.clone()
        self.assertFalse(
            (pt_after_r1[:, 0] == 0).all(), "page_table col 0 should be non-zero"
        )
        # Column 1 should be set (decode_length + last_page_lens = 5 > page_size=4)
        self.assertFalse(
            (pt_after_r1[:, 1] == 0).all(), "page_table col 1 should be non-zero"
        )

        # Replay 2: last_page_lens = 0, decode_length=2 → total 2 tokens → 1 page
        cache_loc_r2 = [
            [16, 17],  # seq 0 topk 0: both on page 4
            [18, 19],  # seq 0 topk 1: both on page 4
            [20, 21],  # seq 1 topk 0: both on page 5
            [22, 23],  # seq 1 topk 1: both on page 5
        ]
        self._call([0, 0], cache_loc_r2)

        # cache_seqlens should be decode_length + 0 = 2 for all seqs
        self.assertTrue(
            (self.cache_seqlens == self.decode_length).all(),
            f"Expected cache_seqlens all={self.decode_length}, got {self.cache_seqlens}",
        )

        # Column 1 MUST be zero now — the fix zeros page_table before scatter
        self.assertTrue(
            (self.page_table[:, 1:] == 0).all(),
            f"Stale page indices found in page_table col 1+ after replay 2: "
            f"{self.page_table}",
        )

    def test_page_table_correct_after_two_replays_growing_seq(self):
        """
        Verify that page_table is correctly populated across two replays when
        sequences grow (the normal decode scenario).
        """
        # Replay 1: all draft tokens on page 0
        cache_loc_r1 = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self._call([0, 0], cache_loc_r1)

        # All draft tokens on page 0 → page_table[:, 0] = 0
        self.assertTrue(
            (self.page_table[:, 0] == 0).all(),
            f"All tokens on page 0, expected 0, got {self.page_table[:, 0]}",
        )

        # Replay 2: sequences grew, last_page_lens=2 now, draft tokens on page 1
        cache_loc_r2 = [[4, 5], [6, 7], [8, 9], [10, 11]]
        self._call([2, 2], cache_loc_r2)

        # cache_seqlens = decode_length + 2 = 4 = page_size → still 1 page
        self.assertTrue(
            (self.cache_seqlens == self.decode_length + 2).all(),
        )
        # col 1 should be zeroed (only 1 page needed)
        self.assertTrue(
            (self.page_table[:, 1:] == 0).all(),
            f"Expected col 1+ zeros, got: {self.page_table}",
        )


if __name__ == "__main__":
    unittest.main()
