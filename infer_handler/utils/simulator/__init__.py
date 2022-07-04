# -*- coding: utf-8 -*-
"""检查器包

提供若干个可以单独运行的工具函数用于检查InferHandler和Observer是否正常工作


"""

from .engine import set_triton_client, set_process_number, _start_engine, infer_image
