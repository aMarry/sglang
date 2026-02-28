import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs


class TestServerArgsWorkarounds(unittest.TestCase):
    @patch("sglang.srt.server_args.envs.SGLANG_ENABLE_SPEC_V2.get", return_value=True)
    def test_switch_decode_backend_for_known_fa3_spec_v2_unstable_combo(self, _):
        args = ServerArgs(model_path="dummy")
        args.disable_cuda_graph = False
        args.attention_backend = "fa3"
        args.decode_attention_backend = None
        args.speculative_algorithm = "EAGLE"
        args.speculative_eagle_topk = 1
        args.page_size = 1
        args.tp_size = 4
        args.dp_size = 1

        args._handle_fa3_spec_v2_cuda_graph_workaround()
        self.assertFalse(args.disable_cuda_graph)
        self.assertEqual(args.decode_attention_backend, "triton")

    @patch("sglang.srt.server_args.envs.SGLANG_ENABLE_SPEC_V2.get", return_value=True)
    def test_keep_decode_backend_for_non_matching_combo(self, _):
        args = ServerArgs(model_path="dummy")
        args.disable_cuda_graph = False
        args.attention_backend = "fa3"
        args.decode_attention_backend = None
        args.speculative_algorithm = "EAGLE"
        args.speculative_eagle_topk = 1
        args.page_size = 1
        args.tp_size = 8
        args.dp_size = 2

        args._handle_fa3_spec_v2_cuda_graph_workaround()
        self.assertFalse(args.disable_cuda_graph)
        self.assertIsNone(args.decode_attention_backend)


if __name__ == "__main__":
    unittest.main()
