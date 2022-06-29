# -*- coding: utf-8 -*-
"""全局变量&相关工具

Function::

    1. get_handler 获取Handler
    2. append_handler 添加Handler

"""
from typing import Type, Optional, Dict

from ._infer_handler import InferHandler

_global_handlers: Dict[str, Type[InferHandler]] = {}
"""全局handler列表"""


def get_handler(handler_name: str) -> Type[InferHandler]:
    """从全局变量中找到目标Handler

    Args:
        handler_name (str): Handler的名字

    Raises:
        ModuleNotFoundError: 找不到相关的模块

    Returns:
        Type[InferHandler]: Handler类
    """
    try:
        return _global_handlers[handler_name]
    except KeyError as e:
        raise ModuleNotFoundError('Handler not found.')


def append_handler(handler_class: Type[InferHandler]) -> Optional[Type[InferHandler]]:
    """添加到全局Handler列表并且自动跳过已经存在的Handler

    Args:
        handler_class (Type[InferHandler]): Handler类

    Returns:
        Optional[Type[InferHandler]]: 成功添加则返回类本身
    """
    # case 只加入不存在的
    if handler_class.name not in _global_handlers:
        # case 初始化成功才加入
        if handler_class.initial_handler():
            _global_handlers[handler_class.name] = handler_class
            _global_handlers[handler_class.module_name] = handler_class
            return handler_class
