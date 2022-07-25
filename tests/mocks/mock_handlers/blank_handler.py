# -*- coding: utf-8 -*-
"""Blank handler for test


"""
from typing import Any, Optional, Iterator

from infer_handler import InferHandler
from infer_handler.structure import HandlerResult, InferTask, ModelResult, TargetObject, Point


class BlankHandler(InferHandler):
    label_nick_name = 'blank'

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[ModelResult]:
        return [
            TargetObject(
                detect_class='blank',
                confidence=0.6,
                box=(Point(0, 0), Point(1, 1))
            ),
            TargetObject(
                detect_class='blank',
                confidence=0.9,
                box=(Point(0, 0), Point(1, 1))
            ),
            TargetObject(
                detect_class='blank',
                confidence=0.9,
                box=(Point(0, 0), Point(1, 1))
            ),
            TargetObject(
                detect_class='blank',
                confidence=0.9,
                box=(Point(0, 0), Point(1, 1))
            ),

        ]
