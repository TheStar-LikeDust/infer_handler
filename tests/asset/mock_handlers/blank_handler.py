# -*- coding: utf-8 -*-
"""Blank handler for test


"""
from typing import Any, Optional

from infer_handler import InferHandler


class BlankHandler(InferHandler):

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return image.shape

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return {
            'shape': kwargs.get('pre_result'),
            'info': 'info_content'
        }
