# -*- coding: utf-8 -*-
"""自动加载

"""

from importlib import import_module

import pkgutil
from logging import getLogger
from typing import NoReturn

from infer_handler import InferHandler, append_handler

logger = getLogger(__name__)
"""日志类"""


def detect_handlers(module_package_path: str) -> NoReturn:
    """自动检测路径下的模块并且将其中的Handler类放入全局Handler列表中

    Args:
        module_package_path (str): 符合Python Packageg格式的路径

    """
    # will raise ModuleNotFound
    target_package = import_module(module_package_path)

    # TODO: detect submodule
    # walk folder
    for walk_item in pkgutil.walk_packages(target_package.__path__):
        # found module name and import target module
        target_module = import_module(f'.{walk_item.name}', package=target_package.__package__)

        for attr_name in dir(target_module):
            # filter the private attr
            if not attr_name.startswith('_'):
                attr = getattr(target_module, attr_name)

                # case: is class and is subclass of InferHandler
                if isinstance(attr, type) and issubclass(attr, InferHandler):
                    # duplication
                    append_handler(attr)

