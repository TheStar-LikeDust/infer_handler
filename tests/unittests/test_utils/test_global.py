import unittest

from infer_handler.utils import _global

from infer_handler import InferHandler
from infer_handler.utils import Observer


class GlobalTestHandler(InferHandler):
    pass


class GlobalTestObserver(Observer):
    pass


class GlobalTestInitialFailedHandler(InferHandler):

    @classmethod
    def _initial_handler(cls):
        assert False


class GlobalTestInitialFailedObserver(Observer):

    def __init__(self, logger=None):
        super().__init__(logger)
        assert False


class UtilsGlobalTestCase(unittest.TestCase):

    def setUp(self) -> None:
        _global._global_observer = []
        _global._global_handlers = []

    def test_get_handler(self):
        """To test_utils.test_detect"""

    def test_append_handler(self):
        """添加Handler"""
        handler = _global.append_handler(GlobalTestHandler)

        assert handler == _global._global_handlers[0]

    def test_append_observer(self):
        """添加Handler"""
        observer = _global.append_observer(GlobalTestObserver)

        assert observer == _global._global_observer[0]

    def test_initial_handler_failed(self):
        """初始化时出错"""
        handler = _global.append_handler(GlobalTestInitialFailedHandler)

        assert handler == None
        assert _global._global_handlers == []

    def test_initial_observer_failed(self):
        """实例化出错"""
        observer = _global.append_observer(GlobalTestInitialFailedObserver)

        assert observer == None
        assert _global._global_observer == []


if __name__ == '__main__':
    unittest.main()
