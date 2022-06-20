import unittest
from concurrent.futures import ProcessPoolExecutor

from infer_handler.utils.detect import detect_handlers
from infer_handler.utils.worker import initial_pool, handler_process

import numpy


class SchedulerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        detect_handlers('tests.asset.mock_handlers')

        initial_pool()

    def test_process_image_handle(self):
        image = numpy.ndarray(shape=(123, 456, 789), dtype=numpy.uint8)

        future = handler_process('blank_handler', image_info=image, other_kwargs={'info': 'info_content'})

        assert future

        result = future.result()

        assert result['info'] == 'info_content'
        assert result['shape'] == (123, 456, 789)


if __name__ == '__main__':
    unittest.main()
