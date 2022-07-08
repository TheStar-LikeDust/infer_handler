import time
import unittest
from concurrent.futures import ProcessPoolExecutor

from infer_handler.utils.detect import detect_handlers
from infer_handler.utils.worker import initial_handler_pool, handler_process

import numpy


def test_initial(number: int):
    print(number)
    pass


class SchedulerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        detect_handlers('tests.asset.mock_handlers')

        initial_handler_pool()

    def test_process_image_handle(self):
        image = numpy.ndarray(shape=(123, 456, 789), dtype=numpy.uint8)

        future = handler_process('BlankHandler', image_info=image, other_kwargs={'info': 'info_content'})

        assert future

        result = future.result()

        assert result['info'] == 'info_content'
        assert result['shape'] == (123, 456, 789)

    def test_initial_handler_pool(self):
        """传入参数"""

        pool = initial_handler_pool(max_worker=2, initial_callback=test_initial, initial_callback_arguments=(5,))

        pool.submit(lambda: None)


if __name__ == '__main__':
    unittest.main()
