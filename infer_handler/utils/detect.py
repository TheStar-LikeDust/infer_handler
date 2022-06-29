# -*- coding: utf-8 -*-
"""自动加载

"""

from importlib import import_module

import pkgutil
from logging import getLogger
from types import ModuleType
from typing import NoReturn, Iterator, Type, List

from infer_handler import InferHandler
from infer_handler._global import _global_handlers, append_handler
from .observer import Observer
from . import _global_observer, append_observer

logger = getLogger(__name__)
"""日志类"""


def _load_module_in_package(module_package_path: str) -> Iterator[ModuleType]:
    """加载并且返回指定package下的所有模块(.py)"""

    # will raise ModuleNotFound
    target_package = import_module(module_package_path)

    for sub_module_info in pkgutil.walk_packages(target_package.__path__):
        yield import_module(f'.{sub_module_info.name}', package=target_package.__package__)


def _find_class_in_module(target_module: ModuleType, target_class: type) -> Iterator[type]:
    """从一个模块中返回一个类的子类"""

    # 过滤私有属性
    for attr_name in dir(target_module):
        if not attr_name.startswith('_'):
            attr = getattr(target_module, attr_name)

            # case: is class and is subclass of target_class
            if isinstance(attr, type) and issubclass(attr, target_class):
                yield attr


def detect_handlers(module_package_path: str) -> NoReturn:
    """自动检测路径下的模块并且将其中的Handler类放入全局Handler列表中

    Args:
        module_package_path (str): 符合Python package格式的路径
    """
    for target_module in _load_module_in_package(module_package_path):

        for handler in _find_class_in_module(target_module, InferHandler):
            append_handler(handler)


def detect_observer(module_package_path: str) -> NoReturn:
    """自动检测路径下的模块并且将其中的observer类放入全局observer列表中

    Args:
        module_package_path (str): 符合Python package格式的路径
    """
    for target_module in _load_module_in_package(module_package_path):

        for observer in _find_class_in_module(target_module, Observer):
            append_observer(observer)


# auto
def auto_detect_handler():
    """自动加载handler"""
    try:
        detect_handlers('handlers')
    except ModuleNotFoundError as e:
        logger.info('Auto detect handlers failed.')


def auto_detect_observer():
    """自动加载observer"""
    try:
        detect_observer('observers')
    except ModuleNotFoundError as e:
        logger.info('Auto detect observer failed.')
