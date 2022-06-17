import sys
import unittest

from infer_handler.utils.detect import detect_handlers

from infer_handler import get_handler


class MyTestCase(unittest.TestCase):
    def test_detect_handlers(self):
        detect_handlers('tests.asset.mock_handlers')

        handler = get_handler('BlankHandler')

        assert handler.name == 'BlankHandler'
        assert handler.module_name == 'blank_handler'


if __name__ == '__main__':
    unittest.main()
