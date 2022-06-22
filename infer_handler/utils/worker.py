# -*- coding: utf-8 -*-
"""多进程运行相关

附带了一个基于ProcessPoolExecutor的多线程处理函数，能够让Python解决一些轻量级的并行推理任务。

相关的函数:

- initial_pool: 初始化进程函数
- handler_process: 执行一次推理任务的入口函数
- infer_callback: 单次推理任务

相关的全局变量:

- pool: 全局进程池
- registered_image_converter: 图像转换器 在子进程的推理任务中处理图像

-------

"""

from concurrent.futures import ProcessPoolExecutor, Future
from typing import Optional, Callable, Dict, Any

from infer_handler import get_handler

pool: Optional[ProcessPoolExecutor] = None
"""核心进程池"""

registered_image_converter: Dict[str, Callable] = {
    'raw_image': lambda x: x,
}
"""已注册的图像转换器字典"""


def initial_pool(initial_callback: Callable = None, initial_callback_arguments: tuple = tuple()):
    """初始化进程

    .. Note::

        在主进程的某些变量可能不会继承到子进程，子进程中需要的函数必须在此方法中执行。

        如：数据库连接、特殊的变量、图像转换器字典、detect_handler等等


    Args:
        initial_callback (Callable, optional): 进程池子进程初始化回调参数. Defaults to None.
        initial_callback_arguments (tuple, optional): 回调函数参数. Defaults to None.
    """
    global pool

    if not initial_callback:
        initial_callback = None
        initial_callback_arguments = tuple()

    pool = ProcessPoolExecutor(initializer=initial_callback, initargs=initial_callback_arguments)


def infer_callback(handle_name: str,
                   image_info: Any,
                   image_converter_name: str,
                   other_kwargs: dict = None) -> Any:
    """单次推理任务

    .. Note:: 运行流程

        1. 根据名字找到某个Handler - 找到某个模型
        2. 处理图片 - 找到图片转换器函数并且处理图片
        3. 推理/处理 - Handler + image + optional(kwargs) = result
        4. TODO: 二次识别


    Args:
        handle_name (str): 模型Handler名字
        image_info (Any): 图片信息（可以是原图也可以是其他自定义的数据）
        image_converter_name (str, optional): 处理图片信息的处理函数名字，需要在初始化是添加到registered_image_processor中. Defaults to 'raw_image'.
        other_kwargs (dict, optional): Handler需要其他的参数. Defaults to None.

    Returns:
        Any: 模型Handler处理的结果
    """
    image_converter_name = image_converter_name if image_converter_name else 'raw_image'
    other_kwargs = other_kwargs if other_kwargs else {}

    # 根据名字找到某个Handler - 找到某个模型
    handle = get_handler(handle_name)

    # 根据传入的名字获取加载图片的处理函数
    image_processor = registered_image_converter.get(image_converter_name)

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
    """执行一次推理任务的入口函数

    在主进程中调用，传入相关信息，子进程将会根据信息执行infer_callback并且返回一个Future。

    """
    return pool.submit(
        infer_callback,
        handle_name,
        image_info,
        image_processor_name,
        other_kwargs,
    )
