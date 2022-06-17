import unittest

from infer_handler import _global

from infer_handler import InferHandler

from infer_handler._global import get_handler, append_handler


class GlobalTest(InferHandler):
    pass


class GlobalTestCase(unittest.TestCase):

    # def test_global_handlers(self):
    #     """全局注册的Handler """
    #     assert _global.__global_handlers == []

    def test_append_and_get(self):
        """添加以及寻找Handler"""

        handler_name = 'GlobalTest'

        with self.subTest('not exist.'):
            with self.assertRaises(ModuleNotFoundError) as e:
                get_handler(handler_name)

        with self.subTest('append handler'):
            append_handler(GlobalTest)

        with self.subTest('get handler'):
            handler = get_handler(handler_name)

            assert handler == GlobalTest


if __name__ == '__main__':
    unittest.main()
