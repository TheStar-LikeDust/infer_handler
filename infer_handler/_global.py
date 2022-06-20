# -*- coding: utf-8 -*-
"""全局变量&相关工具

Function::

    1. get_handler 获取Handler
    2. append_handler 添加Handler

"""
from typing import List, Type, Optional

from ._infer_handler import InferHandler

__global_handlers: List[Type[InferHandler]] = []
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
    for handler in __global_handlers:
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
    if handler_class not in __global_handlers:
        # case 初始化成功才加入
        if handler_class.initial_handler():
            __global_handlers.append(handler_class)
            return handler_class
