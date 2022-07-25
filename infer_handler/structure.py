# -*- coding: utf-8 -*-
"""Common typing alies.

"""
from typing import NamedTuple, Any, Sequence, TypedDict, Union, Tuple, List, Dict


class Point(NamedTuple):
    x: Union[float, int]
    y: Union[float, int]


Points = Sequence[Point]


class InferTask(NamedTuple):
    handle_name: str
    image_info: Any
    image_converter_name: str = ''
    sub_handlers: Sequence[str] = []
    parameter: dict = {}


class TargetObject(TypedDict, total=False):
    detect_class: Union[str, int]
    confidence: float
    box: Tuple[Point, Point]


class ClassObject(TypedDict, total=False):
    detect_class: Union[str, int]
    confidence: float


ModelResult = List[Union[TargetObject, ClassObject]]
"""后处理结果"""

HandlerResult = Dict[str, ModelResult]
