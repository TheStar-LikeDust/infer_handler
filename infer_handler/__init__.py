# -*- coding: utf-8 -*-
"""infer handler.


"""
from infer_handler._infer_handler import InferHandler

__version__ = '2022.7.25'

TRITON_CLIENT_HTTP_FLAG = False


def switch_to_http():
    global TRITON_CLIENT_HTTP_FLAG
    TRITON_CLIENT_HTTP_FLAG = True
