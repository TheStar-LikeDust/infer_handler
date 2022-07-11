import unittest

from tritonclient.grpc import InferenceServerClient as grpc_client

from tritonclient.http import InferenceServerClient as http_client

from infer_handler import switch_to_http


class TritonHandlerTestCase(unittest.TestCase):

    @unittest.skip
    def test_switch_flag(self):
        switch_to_http()

        from infer_handler.triton_handler import InferenceServerClient

        assert InferenceServerClient is http_client

    def test_infer(self):
        """TODO"""


if __name__ == '__main__':
    unittest.main()
