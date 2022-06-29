# -*- coding: utf-8 -*-
"""Example Google style docstrings.


"""
from typing import Any, NoReturn, Optional

from infer_handler.utils import Observer


class Normal(Observer):
    required_models = ['normal']

    def judge(self, result: Any) -> Optional[bool]:
        print('judge!')

    def trigger(self) -> NoReturn:
        pass
