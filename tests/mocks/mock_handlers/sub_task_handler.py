# -*- coding: utf-8 -*-
"""Example Google style docstrings.

"""
from typing import Any, Iterator, Optional, Tuple

from infer_handler import InferHandler
from infer_handler.structure import HandlerResult, InferTask, ClassObject, ModelResult


class SubBlankHandler(InferHandler):

    @classmethod
    def _process_sub_task(cls, image: Any, infer_result: HandlerResult) -> Iterator[Tuple[InferTask, dict]]:
        for blank_item in infer_result.get('blank'):
            yield InferTask(handle_name='SubBlankHandler', image_info=None), blank_item

    @classmethod
    def _initial_handler(cls):
        pass

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[ModelResult]:
        return [
            ClassObject(
                detect_class='sub',
                confidence=0.75,
            )
        ]
