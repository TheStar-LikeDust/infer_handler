# -*- coding: utf-8 -*-
"""infer handler.


"""

__version__ = '2022.7.11'

TRITON_CLIENT_HTTP_FLAG = False


def switch_to_http():
    global TRITON_CLIENT_HTTP_FLAG
    TRITON_CLIENT_HTTP_FLAG = True


from ._infer_handler import InferHandler
