# -*- coding: utf-8 -*-
"""基于Triton的InferHandler

"""

# TODO: switch http or grpc
from typing import Any, Optional, List

from tritonclient.http import InferInput, InferRequestedOutput, InferenceServerClient, InferAsyncRequest

from . import InferHandler

CLIENT: Optional[InferenceServerClient] = None
"""当前全局的Triton 客户端连接"""


def get_client() -> InferenceServerClient:
    """获取当前Triton客户端"""
    return CLIENT


def set_client(client: InferenceServerClient) -> InferenceServerClient:
    """设置当前Triton客户端，子进程初始化的时候使用"""
    global CLIENT
    CLIENT = client
    return CLIENT


class TritonHandler(InferHandler):
    """基于Triton的ImageHandler类，使用Python Backend包装前后处理

    .. Note::
        Triton所需要的Input、Output尽可能置为全局变量，方便子进程初始化

    """

    triton_model_name: str = None
    """Triton中的模型别名"""

    triton_inputs: List[InferInput] = None
    """Triton需要的输入格式"""

    triton_outputs: List[InferRequestedOutput] = None
    """Triton需要的输出格式"""

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional:
        """默认的Triton处理流程"""
        client = get_client()

        # TODO: 异步
        infer_result = client.infer(
            model_name=cls.triton_model_name,
            inputs=cls.triton_inputs,
            outputs=cls.triton_outputs,
            # timeout=1
        )

        return infer_result
