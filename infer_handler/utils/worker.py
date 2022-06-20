# -*- coding: utf-8 -*-
"""多线程调度器模块

"""

from concurrent.futures import ProcessPoolExecutor, Future
from typing import Optional, Callable, Dict, Any

from infer_handler import get_handler

# TODO: lazy import
try:
    # from shared_memory_toolkit import dump_image_into_shared_memory, load_image_from_shared_memory
    from numpy import ndarray
except ModuleNotFoundError as e:
    raise e

pool: Optional[ProcessPoolExecutor] = None
"""核心进程池"""

registered_image_processor: Dict[str, Callable] = {
    'raw_image': lambda x: x,
    # 'shared_memory': lambda x: load_image_from_shared_memory(x),
}


def initial_pool(initial_callback: Callable = None, initial_callback_arguments: tuple = tuple()):
    """TODO: 进程池

    Args:
        initial_callback (Callable, optional): 进程池子进程初始化回调参数. Defaults to None.
        initial_callback_arguments (tuple, optional): 回调函数参数. Defaults to None.
    """
    # TODO: callback
    global pool

    if not initial_callback:
        initial_callback = None
        initial_callback_arguments = tuple()

    pool = ProcessPoolExecutor(initializer=initial_callback, initargs=initial_callback_arguments)


def infer_callback(handle_name: str,
                   image_info: Any,
                   image_processor_name: str,
                   other_kwargs: dict = None) -> Any:
    """共享内存的一次推理任务函数

    Args:
        handle_name (str): 模型Handler名字
        image_info (Any): 图片信息（可以是原图也可以是其他自定义的数据）
        image_processor_name (str, optional): 处理图片信息的处理函数名字，需要在初始化是添加到registered_image_processor中. Defaults to 'raw_image'.
        other_kwargs (dict, optional): Handler需要其他的参数. Defaults to None.

    Returns:
        Any: 模型Handler处理的结果
    """
    image_processor_name = image_processor_name if image_processor_name else 'raw_image'
    other_kwargs = other_kwargs if other_kwargs else {}

    # 根据名字找到某个Handler - 找到某个模型
    handle = get_handler(handle_name)

    # 根据传入的名字获取加载图片的处理函数
    image_processor = registered_image_processor.get(image_processor_name)

    # 通过图片处理函数处理图片
    image = image_processor(image_info)

    # 图片 + 模型处理 + （可选的参数） = 推理结果
    handle_result = handle.image_handle(image, **other_kwargs)

    return handle_result


def handler_process(handle_name: str,
                    image_info: Any = object(),
                    image_processor_name: str = 'raw_image',
                    other_kwargs: dict = None
                    ) -> Future:
    """添加任务的入口函数"""
    return pool.submit(
        infer_callback,
        handle_name,
        image_info,
        image_processor_name,
        other_kwargs,
    )
