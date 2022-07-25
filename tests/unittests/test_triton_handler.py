import unittest

from tritonclient.grpc import InferenceServerClient as grpc_client

from tritonclient.http import InferenceServerClient as http_client

from infer_handler import switch_to_http
from infer_handler.utils import detect_handlers
from infer_handler.utils._global import get_handler


class TritonHandlerTestCase(unittest.TestCase):

    @unittest.skip
    def test_switch_flag(self):
        switch_to_http()

        from infer_handler.triton_handler import InferenceServerClient

        assert InferenceServerClient is http_client

    def test_get_client(self):
        """test get client"""


if __name__ == '__main__':
    unittest.main()
