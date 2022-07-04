import time
import unittest

import numpy

from infer_handler.utils import *


def initial_callback():
    detect_handlers('tests.asset.mock_handlers')


pic = numpy.ndarray((1080, 1920, 3), dtype=numpy.uint8)


class MyTestCase(unittest.TestCase):

    def test_atom(self):
        """单一任务"""
        detect_observer('tests.asset.mock_observer')

        # initial process pool
        initial_handler_pool(8, initial_callback, )
        # initial thread pool
        initial_observer_pool()

        # generate infer task

        f = handler_process('blank_handler', pic)
        f.add_done_callback(lambda x: observer_process('normal', x.result()))

        time.sleep(1)

    def test_sequence(self):
        """多个任务"""
        pass


if __name__ == '__main__':
    unittest.main()
