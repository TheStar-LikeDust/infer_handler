# -*- coding: utf-8 -*-
"""Example Google style docstrings.

"""
import time
from typing import Any, Iterator, Optional, Tuple

from infer_handler import InferHandler
from infer_handler.structure import HandlerResult, InferTask, ClassObject, ModelResult


class SubErrorHandler(InferHandler):

    @classmethod
    def _process_sub_task(cls, image: Any, infer_result: HandlerResult) -> Iterator[Tuple[InferTask, dict]]:
        for blank_item in infer_result.get('blank'):
            yield InferTask(handle_name=cls.name, image_info=None), blank_item

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        assert False

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[ModelResult]:
        assert False
