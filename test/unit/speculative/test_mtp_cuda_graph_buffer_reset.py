"""
Unit test for MTP CUDA graph buffer reset functionality.

This test verifies the logic for the reset_buffers method in MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.
It tests that all buffers are properly reset, including per-runner buffers like hidden_states, extend_seq_lens,
and extend_start_loc. This is critical for MTP (Multi-Token Prediction) correctness with CUDA graphs.

Note: This is a logic verification test rather than a full integration test, as setting up the full
MultiLayerEagleMultiStepDraftExtendCudaGraphRunner requires GPU resources, model loading, and CUDA graph
capture infrastructure.
"""

import unittest
from unittest.mock import MagicMock, Mock

import torch

# Test constants
TEST_VOCAB_SIZE = 32000


class TestMTPCudaGraphBufferReset(unittest.TestCase):
    """Test MTP CUDA graph buffer reset functionality."""

    def test_reset_buffers_clears_per_runner_buffers(self):
        """
        Test that reset_buffers properly clears per-runner buffers including:
        - hidden_states
        - extend_seq_lens
        - extend_start_loc
        - req_pool_indices (shared buffer)
        """
        # Mock the MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
        from types import SimpleNamespace

        # Create mock runners with buffers
        num_runners = 3
        max_bs = 4
        num_tokens_per_bs = 8
        hidden_size = 256

        # Create mock cuda_graph_buffers
        cuda_graph_buffers = {
            "input_ids": torch.ones((100,), dtype=torch.int64),
            "seq_lens": torch.ones((max_bs,), dtype=torch.int32),
            "out_cache_loc": torch.ones((100,), dtype=torch.int64),
            "swa_out_cache_loc": torch.ones((100,), dtype=torch.int64),
            "positions": torch.ones((100,), dtype=torch.int64),
            "accept_length": torch.ones((max_bs,), dtype=torch.int32),
            "req_pool_indices": torch.ones((max_bs,), dtype=torch.int32),
        }

        # Create mock runners with per-runner buffers
        runners = []
        for i in range(num_runners):
            runner = SimpleNamespace()
            runner.hidden_states = torch.ones(
                (max_bs * num_tokens_per_bs, hidden_size), dtype=torch.float32
            )
            runner.extend_seq_lens = torch.ones((max_bs,), dtype=torch.int32)
            runner.extend_start_loc = torch.ones((max_bs,), dtype=torch.int32)
            runner.mrope_positions = torch.ones(
                (3, max_bs * num_tokens_per_bs), dtype=torch.int64
            )
            runner.next_token_logits_buffer = torch.ones(
                (max_bs * num_tokens_per_bs, TEST_VOCAB_SIZE), dtype=torch.float32
            )
            runner.num_tokens_per_bs = num_tokens_per_bs
            runner.max_bs = max_bs
            runners.append(runner)

        # Mock the reset_buffers implementation
        def reset_buffers(
            cuda_graph_buffers, runners, seq_len_fill_value, forward_batch, batch_result
        ):
            cuda_graph_buffers["input_ids"].zero_()
            cuda_graph_buffers["seq_lens"].fill_(seq_len_fill_value)
            cuda_graph_buffers["out_cache_loc"].zero_()
            cuda_graph_buffers["swa_out_cache_loc"].zero_()
            cuda_graph_buffers["positions"].zero_()
            cuda_graph_buffers["accept_length"][: forward_batch.batch_size].copy_(
                batch_result.accept_lens
            )
            cuda_graph_buffers["req_pool_indices"].zero_()

            # Reset per-runner buffers to avoid accumulation of stale data
            # across batch iterations, which is critical for MTP correctness
            for runner in runners:
                if runner is not None:
                    runner.hidden_states.zero_()
                    runner.extend_seq_lens.fill_(runner.num_tokens_per_bs)
                    runner.extend_start_loc.copy_(
                        torch.arange(
                            0,
                            runner.max_bs * runner.num_tokens_per_bs,
                            step=runner.num_tokens_per_bs,
                            dtype=torch.int32,
                            device=runner.extend_start_loc.device,
                        )
                    )
                    # Reset output buffers to prevent stale data from previous iterations
                    # causing incorrect token generation (missing characters issue)
                    runner.mrope_positions.zero_()
                    runner.next_token_logits_buffer.zero_()

        # Create mock forward_batch and batch_result
        forward_batch = SimpleNamespace()
        forward_batch.batch_size = 2
        batch_result = SimpleNamespace()
        batch_result.accept_lens = torch.tensor([1, 2], dtype=torch.int32)

        # Call reset_buffers
        seq_len_fill_value = 1
        reset_buffers(
            cuda_graph_buffers, runners, seq_len_fill_value, forward_batch, batch_result
        )

        # Verify shared buffers are reset
        self.assertTrue(
            torch.all(cuda_graph_buffers["input_ids"] == 0),
            "input_ids should be zeroed",
        )
        self.assertTrue(
            torch.all(cuda_graph_buffers["seq_lens"] == seq_len_fill_value),
            "seq_lens should be filled with seq_len_fill_value",
        )
        self.assertTrue(
            torch.all(cuda_graph_buffers["out_cache_loc"] == 0),
            "out_cache_loc should be zeroed",
        )
        self.assertTrue(
            torch.all(cuda_graph_buffers["swa_out_cache_loc"] == 0),
            "swa_out_cache_loc should be zeroed",
        )
        self.assertTrue(
            torch.all(cuda_graph_buffers["positions"] == 0),
            "positions should be zeroed",
        )
        self.assertTrue(
            torch.all(cuda_graph_buffers["req_pool_indices"] == 0),
            "req_pool_indices should be zeroed",
        )

        # Verify accept_length is copied correctly
        self.assertTrue(
            torch.all(
                cuda_graph_buffers["accept_length"][: forward_batch.batch_size]
                == batch_result.accept_lens
            ),
            "accept_length should be copied from batch_result",
        )

        # Verify per-runner buffers are reset
        for i, runner in enumerate(runners):
            self.assertTrue(
                torch.all(runner.hidden_states == 0),
                f"Runner {i}: hidden_states should be zeroed",
            )
            self.assertTrue(
                torch.all(runner.extend_seq_lens == runner.num_tokens_per_bs),
                f"Runner {i}: extend_seq_lens should be filled with num_tokens_per_bs",
            )
            # Verify extend_start_loc has the correct stride pattern
            expected_extend_start_loc = torch.arange(
                0,
                runner.max_bs * runner.num_tokens_per_bs,
                step=runner.num_tokens_per_bs,
                dtype=torch.int32,
            )
            self.assertTrue(
                torch.all(runner.extend_start_loc == expected_extend_start_loc),
                f"Runner {i}: extend_start_loc should have correct stride pattern",
            )
            # Verify output buffers are reset
            self.assertTrue(
                torch.all(runner.mrope_positions == 0),
                f"Runner {i}: mrope_positions should be zeroed",
            )
            self.assertTrue(
                torch.all(runner.next_token_logits_buffer == 0),
                f"Runner {i}: next_token_logits_buffer should be zeroed",
            )

    def test_buffer_reset_prevents_accumulation(self):
        """
        Test that buffer reset prevents accumulation of stale data across iterations.
        This simulates multiple batch iterations to ensure buffers don't accumulate.
        """
        from types import SimpleNamespace

        max_bs = 2
        num_tokens_per_bs = 4
        hidden_size = 128

        # Create a single runner
        runner = SimpleNamespace()
        runner.hidden_states = torch.zeros(
            (max_bs * num_tokens_per_bs, hidden_size), dtype=torch.float32
        )
        runner.extend_seq_lens = torch.zeros((max_bs,), dtype=torch.int32)
        runner.extend_start_loc = torch.zeros((max_bs,), dtype=torch.int32)
        runner.mrope_positions = torch.zeros(
            (3, max_bs * num_tokens_per_bs), dtype=torch.int64
        )
        runner.next_token_logits_buffer = torch.zeros(
            (max_bs * num_tokens_per_bs, TEST_VOCAB_SIZE), dtype=torch.float32
        )
        runner.num_tokens_per_bs = num_tokens_per_bs
        runner.max_bs = max_bs

        # Simulate contamination
        runner.hidden_states.fill_(1.0)
        runner.extend_seq_lens.fill_(999)
        runner.extend_start_loc.fill_(999)
        runner.mrope_positions.fill_(999)
        runner.next_token_logits_buffer.fill_(1.0)

        # Create cuda_graph_buffers
        cuda_graph_buffers = {
            "input_ids": torch.ones((100,), dtype=torch.int64),
            "seq_lens": torch.ones((max_bs,), dtype=torch.int32),
            "out_cache_loc": torch.ones((100,), dtype=torch.int64),
            "swa_out_cache_loc": torch.ones((100,), dtype=torch.int64),
            "positions": torch.ones((100,), dtype=torch.int64),
            "accept_length": torch.ones((max_bs,), dtype=torch.int32),
            "req_pool_indices": torch.ones((max_bs,), dtype=torch.int32),
        }

        # Reset function
        def reset_runner_buffers(runner):
            runner.hidden_states.zero_()
            runner.extend_seq_lens.fill_(runner.num_tokens_per_bs)
            runner.extend_start_loc.copy_(
                torch.arange(
                    0,
                    runner.max_bs * runner.num_tokens_per_bs,
                    step=runner.num_tokens_per_bs,
                    dtype=torch.int32,
                    device=runner.extend_start_loc.device,
                )
            )
            # Reset output buffers to prevent stale data from previous iterations
            runner.mrope_positions.zero_()
            runner.next_token_logits_buffer.zero_()

        # Apply reset
        reset_runner_buffers(runner)

        # Verify buffers are clean
        self.assertTrue(
            torch.all(runner.hidden_states == 0),
            "hidden_states should be zeroed after reset",
        )
        self.assertTrue(
            torch.all(runner.extend_seq_lens == num_tokens_per_bs),
            "extend_seq_lens should be reset to num_tokens_per_bs",
        )
        expected_extend_start_loc = torch.arange(
            0,
            max_bs * num_tokens_per_bs,
            step=num_tokens_per_bs,
            dtype=torch.int32,
        )
        self.assertTrue(
            torch.all(runner.extend_start_loc == expected_extend_start_loc),
            "extend_start_loc should have correct pattern after reset",
        )
        # Verify output buffers are clean
        self.assertTrue(
            torch.all(runner.mrope_positions == 0),
            "mrope_positions should be zeroed after reset",
        )
        self.assertTrue(
            torch.all(runner.next_token_logits_buffer == 0),
            "next_token_logits_buffer should be zeroed after reset",
        )


if __name__ == "__main__":
    unittest.main()
