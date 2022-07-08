import sys
import unittest

from infer_handler.utils.detect import detect_handlers

from infer_handler.utils._global import get_handler


class MyTestCase(unittest.TestCase):
    def test_detect_handlers(self):
        """get_handler"""
        detect_handlers('tests.asset.mock_handlers')

        handler = get_handler('BlankHandler')

        assert handler.name == 'BlankHandler'

    def test_not_exist_handler(self):
        """get_handler if handler not exist"""

        with self.assertRaises(ModuleNotFoundError) as e:
            get_handler('omega')

    def test_handler_process_not_exist(self):
        """handler_process if handler not exist"""
        with self.assertRaises(Exception) as e:
            from infer_handler.utils import handler_process, initial_handler_pool
            initial_handler_pool()
            f = handler_process('omega', object)
            f.result()


if __name__ == '__main__':
    unittest.main()
