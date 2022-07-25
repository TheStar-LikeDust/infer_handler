import time
import unittest
from typing import NoReturn, Optional

from infer_handler.utils.observer import Observer


class Mock(Observer):
    name = 'Mock'

    required_models = ['mock', ]

    def judge(self) -> Optional[bool]:
        return True

    alarm_flag = False

    def trigger(self) -> NoReturn:
        self.last_trigger_time = time.time()

        self.alarm_flag = True


class ObserverTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from infer_handler.utils.worker import initial_observer_pool, observer_process

        initial_observer_pool()

    def setUp(self) -> None:
        from infer_handler.utils import _global
        # clear
        _global._global_observer = []

    def test_judge_manual(self):
        """observer judge callback"""
        mock = Mock()

        res_list = [i for i in range(20)]

        [mock.observer_judge_callback('mock', _) for _ in res_list]

        assert mock.alarm_flag == True


if __name__ == '__main__':
    unittest.main()
