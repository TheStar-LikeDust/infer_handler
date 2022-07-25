# -*- coding: utf-8 -*-
"""常用工具

"""
from .observer import Observer

# detect
from .detect import detect_handlers, detect_observer, auto_detect_observer, auto_detect_handler

from ._global import get_handler

# worker
from .worker import initial_handler_pool, initial_observer_pool, _handler_process, handler_process, observer_process

# Auto detect observers.
auto_detect_observer()
