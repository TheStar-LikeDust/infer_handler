import time
import unittest
from infer_handler import switch_to_http

switch_to_http()

import os

from multiprocessing import current_process, parent_process, active_children

from infer_handler.utils.detect import detect_handlers
from infer_handler.utils.worker import initial_handler_pool, handler_process
from infer_handler.structure import InferTask

import numpy


class SchedulerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        detect_handlers('tests.mocks.mock_handlers')
        initial_handler_pool(max_worker=1)

    @classmethod
    def tearDownClass(cls) -> None:
        [os.kill(p.pid, 7) for p in active_children()]

    def test_process_image_handle(self):
        handler_result = handler_process(InferTask(
            handle_name='BlankHandler',
            image_info=None,
            parameter={'info': 'info_content'}
        )).result()

        blank_items = handler_result.get('blank')

        assert len(blank_items) == 4

    def test_handler_process_failed(self):
        """Failed"""
        handler_result = handler_process(InferTask(
            handle_name='SubErrorHandler',
            image_info=None,
        )).result()



    def test_sub_task(self):
        """case: sub-task."""

        image = numpy.ndarray(shape=(123, 456, 789), dtype=numpy.uint8)

        handler_result = handler_process(InferTask(
            handle_name='BlankHandler',
            image_info=image,
            sub_handlers=['SubBlankHandler'],
            parameter={'info': 'info_content'}
        )).result()

        blank_items = handler_result.get('blank')

        assert blank_items[0].get('SubBlankHandler')

    def test_sub_task_failed(self):
        """case: sub-task failed."""

        image = numpy.ndarray(shape=(123, 456, 789), dtype=numpy.uint8)

        handler_result = handler_process(InferTask(
            handle_name='BlankHandler',
            image_info=image,
            sub_handlers=['SubErrorHandler'],
        )).result()

        blank_items = handler_result.get('blank')
        print(blank_items)

    def test_sub_task_timeout(self):
        """case: sub-task timeout."""

        image = numpy.ndarray(shape=(123, 456, 789), dtype=numpy.uint8)

        handler_result = handler_process(InferTask(
            handle_name='BlankHandler',
            image_info=image,
            sub_handlers=['SubTimeoutHandler'],
        )).result()

    def test_handler_process_timecost(self):
        """process A blank InferTask in BlankHandler"""

        handler_result = handler_process(InferTask(
            handle_name='BlankHandler',
            image_info=None,
        )).result()

        with self.subTest('blank InferTask'):
            s = time.time()
            handler_result = handler_process(InferTask(
                handle_name='BlankHandler',
                image_info=None,
            )).result()
            print('cost', time.time() - s)

        with self.subTest('raw image InferTask'):
            random_image = numpy.ndarray((1080, 1920, 3), dtype=numpy.uint8)
            s = time.time()
            handler_result = handler_process(InferTask(
                handle_name='BlankHandler',
                image_info=random_image,
            )).result()
            print('transfer image cost', time.time() - s)

        with self.subTest('raw image InferTask'):
            from shared_memory_toolkit import dump_image_into_shared_memory
            random_image = numpy.ndarray((1080, 1920, 3), dtype=numpy.uint8)
            dump_image_into_shared_memory('random_image', random_image)

            s = time.time()
            handler_result = handler_process(InferTask(
                handle_name='BlankHandler',
                image_info='random_image',
                image_converter_name='shm'
            )).result()
            print('shm cost', time.time() - s)


if __name__ == '__main__':
    unittest.main()
