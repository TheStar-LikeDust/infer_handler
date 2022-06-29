# -*- coding: utf-8 -*-
"""常用工具

"""
from typing import List, Type

from .observer import Observer

# global
_global_observer: List[Observer] = []


def append_observer(observer_class: Type[Observer]):
    if observer_class not in [_.__class__ for _ in _global_observer]:
        _global_observer.append(observer_class())


# detect
from .detect import detect_handlers, detect_observer, auto_detect_observer

# worker
from .worker import initial_handler_pool, initial_observer_pool, handler_process, observer_process

auto_detect_observer()
