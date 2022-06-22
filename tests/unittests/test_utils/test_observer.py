import time
import unittest
from typing import Any, NoReturn

from infer_handler.utils.observer import Observer, observer_judge_callback


class Mock(Observer):
    name = 'Mock'

    def judge(self, result: Any) -> bool:
        return result >= 100

    alarm_flag = False

    def trigger(self) -> NoReturn:
        self.last_trigger_time = time.time()

        self.alarm_flag = True


class MyTestCase(unittest.TestCase):
    def test_trigger(self):
        mock = Mock()

        res_list = [i for i in range(20)]

        [observer_judge_callback(mock, _) for _ in res_list]

        assert mock.alarm_flag == True


if __name__ == '__main__':
    unittest.main()
