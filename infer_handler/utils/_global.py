# -*- coding: utf-8 -*-
"""全局变量&相关工具

Function::

    1. get_handler 获取Handler
    2. append_handler 添加Handler

"""
from typing import Type, Optional, Dict, List

from infer_handler._infer_handler import InferHandler
from infer_handler.utils import Observer

_global_handlers: List[Type[InferHandler]] = []
"""全局handler列表"""

_global_observer: List[Observer] = []
"""全局Observers列表"""

# TODO: need cache?

def get_handler(handler_name: str) -> Type[InferHandler]:
    """从全局变量中找到目标Handler

    Args:
        handler_name (str): Handler的名字

    Raises:
        ModuleNotFoundError: 找不到相关的模块

    Returns:
        Type[InferHandler]: Handler类
    """
    for handler in _global_handlers:
        if handler.name == handler_name or handler.module_name == handler_name:
            return handler
    raise ModuleNotFoundError('Handler not found.')


def append_handler(handler_class: Type[InferHandler]) -> Optional[Type[InferHandler]]:
    """添加到全局Handler列表并且自动跳过已经存在的Handler

    Args:
        handler_class (Type[InferHandler]): Handler类

    Returns:
        Optional[Type[InferHandler]]: 成功添加则返回类本身
    """
    # case 只加入不存在的
    if handler_class not in _global_handlers:
        # case 初始化成功才加入
        if handler_class.initial_handler():
            _global_handlers.append(handler_class)
            return handler_class


def append_observer(observer_class: Type[Observer]):
    """添加到全局Observers列表并且自动跳过已存在的Observer"""
    if observer_class not in [_.__class__ for _ in _global_observer] and observer_class is not Observer:
        _global_observer.append(observer_class())
