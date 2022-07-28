# -*- coding: utf-8 -*-
"""Example Google style docstrings.


"""
from typing import Any, NoReturn, Optional

from infer_handler.utils import Observer


class Normal(Observer):
    required_labels = ['blank']

    def judge(self) -> Optional[bool]:
        print('judge!')

    def trigger(self) -> NoReturn:
        pass
