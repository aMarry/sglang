import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import dp_attention


class TestDpAttentionInit(unittest.TestCase):
    @patch("sglang.srt.layers.dp_attention.get_tensor_model_parallel_world_size")
    @patch("sglang.srt.layers.dp_attention.get_tensor_model_parallel_rank")
    def test_disable_dp_attention_when_dp_size_is_one(self, mock_tp_rank, mock_tp_size):
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 4

        server_args = SimpleNamespace(
            enable_dp_attention=True,
            dp_size=1,
            moe_dense_tp_size=None,
            attn_cp_size=1,
            device="cpu",
        )
        model_config = SimpleNamespace(hidden_size=16, dtype=torch.float16)

        dp_attention.initialize_dp_attention(server_args, model_config)
        self.assertFalse(dp_attention.is_dp_attention_enabled())


if __name__ == "__main__":
    unittest.main()
