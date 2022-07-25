# -*- coding: utf-8 -*-
"""基于Triton的InferHandler

"""
from threading import Event
from typing import Any, Optional, List, Callable, Tuple
from types import FunctionType

from tritonclient.http import InferenceServerClient as http_client
from tritonclient.grpc import InferenceServerClient as grpc_client

from . import InferHandler, TRITON_CLIENT_HTTP_FLAG

try:
    if TRITON_CLIENT_HTTP_FLAG:
        from tritonclient.http import InferInput, InferRequestedOutput, InferenceServerClient, InferAsyncRequest
    else:
        from tritonclient.grpc import InferInput, InferRequestedOutput, InferenceServerClient, InferResult, \
            InferenceServerException
except ModuleNotFoundError as e:
    print('cannot import tritonclient', e)

TRITON_SERVER_HOST: str = 'localhost'
"""Triton server host"""

TRITON_SERVER_PORT: int = 8000
"""Triton server port"""

_client: Optional[InferenceServerClient] = None
"""当前全局的Triton 客户端连接"""


def get_client() -> InferenceServerClient:
    """获取当前Triton客户端"""
    global _client
    if not _client:
        _client = InferenceServerClient(url=f'{TRITON_SERVER_HOST}:{TRITON_SERVER_PORT}')
    assert _client.is_server_ready(), 'Triton server not ready.'
    return _client


def set_client(host: str, port: int):
    """设置连接"""
    global TRITON_SERVER_HOST, TRITON_SERVER_PORT
    TRITON_SERVER_HOST = host
    TRITON_SERVER_PORT = port


# def set_client(client: InferenceServerClient) -> InferenceServerClient:
#     """设置当前Triton客户端，子进程初始化的时候使用"""
#     global _client
#     _client = _client
#     return _client


def grpc_callback() -> Tuple[Event, list, Callable]:
    """generate callback for grpc client."""
    event = Event()
    event.clear()

    infer_result_handle = []

    def inner_callback(result: InferResult, error: InferenceServerException):
        # TODO: error output
        infer_result_handle.append(result)

        event.set()

    return event, infer_result_handle, inner_callback


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

        if isinstance(client, http_client):
            client: http_client

            infer_result: InferAsyncRequest = client.async_infer(
                model_name=cls.triton_model_name,
                inputs=cls.triton_inputs,
                outputs=cls.triton_outputs,
                timeout=1,
            )

            return infer_result.get_result(block=True, timeout=1)
        else:
            client: grpc_client

            event, infer_result_handle, callback = grpc_callback()

            client.async_infer(
                model_name=cls.triton_model_name,
                inputs=cls.triton_inputs,
                outputs=cls.triton_outputs,
                client_timeout=1,
                callback=callback,
            )

            event.wait(1.1)

            # TODO: error
            assert infer_result_handle[0], 'error'
            return infer_result_handle[0]
