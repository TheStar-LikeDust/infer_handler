import time
import unittest

import numpy

from infer_handler.structure import InferTask
from infer_handler.utils import *


def initial_callback():
    detect_handlers('tests.mocks.mock_handlers')


pic = numpy.ndarray((1080, 1920, 3), dtype=numpy.uint8)


class MyTestCase(unittest.TestCase):

    def test_atom(self):
        """单一任务"""
        detect_observer('tests.mocks.mock_observer')

        # initial process pool
        initial_handler_pool(8, initial_callback, )
        # initial thread pool
        initial_observer_pool()

        # generate infer task
        f = handler_process(
            InferTask(
                handle_name='BlankHandler',
                image_info=None,
            )
        )
        f.add_done_callback(lambda x: observer_process(x.result()))

        time.sleep(1)

    def test_sequence(self):
        """多个任务"""
        pass


if __name__ == '__main__':
    unittest.main()
