# -*- coding: utf-8 -*-
"""全局变量&相关工具

管理全局的InferHandler和Observe

函数:

    1. get_handler: 通过类名获取Handler
    2. append_handler: 添加InferHandler 带过滤
    3. append_observer: 添加Observer 带过滤且会无参数实例化

TODO:

    是否需要做字典cache?

"""
from logging import getLogger
from typing import Type, Optional, List

from infer_handler._infer_handler import InferHandler
from infer_handler.utils import Observer

logger = getLogger('common')

_global_handlers: List[Type[InferHandler]] = []
"""全局handler列表"""

_global_observer: List[Observer] = []
"""全局Observers列表"""


def get_handler(handler_name: str) -> Type[InferHandler]:
    """从全局变量中找到目标Handler

    如果不存在，则会抛出ModuleNotFoundError

    Args:
        handler_name (str): Handler的名字

    Raises:
        ModuleNotFoundError: 找不到相关的模块

    Returns:
        Type[InferHandler]: Handler类
    """
    for handler in _global_handlers:
        if handler.name == handler_name == handler_name:
            return handler
    raise ModuleNotFoundError('Handler not found.')


def append_handler(handler_class: Type[InferHandler]) -> Optional[Type[InferHandler]]:
    """添加到全局Handler列表并且自动跳过已经存在的Handler

    只有成功添加才会返回添加的类本身

    Args:
        handler_class (Type[InferHandler]): Handler类

    Returns:
        Optional[Type[InferHandler]]: 成功添加的类
    """
    # case 只加入不存在的
    if handler_class not in _global_handlers:
        # case 初始化成功才加入
        if handler_class.initial_handler():
            _global_handlers.append(handler_class)
            return handler_class


def append_observer(observer_class: Type[Observer]) -> Optional[Observer]:
    """添加到全局Observers列表并且自动跳过已存在的Observer

    同时会实例化Observer类

    只有成功添加才会返回添加的实例自身

    Args:
        observer_class (Type[Observer]): Observer类

    Returns:
        Optional[Type[InferHandler]]: 成功添加的实例
    """
    if observer_class not in [_.__class__ for _ in _global_observer] and observer_class is not Observer:
        try:
            ins = observer_class()
        except Exception as e:
            logger.error(f'添加Observer: {observer_class}出错', exc_info=e)
        else:
            _global_observer.append(ins)
            return ins
