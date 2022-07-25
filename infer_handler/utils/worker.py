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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from threading import Semaphore
from typing import Optional, Callable, Dict, Any, List, Tuple

from .detect import auto_detect_handler
from ._global import _global_observer, get_handler

from shared_memory_toolkit import load_image_from_shared_memory

# structure
from ..structure import InferTask, HandlerResult

process_pool: Optional[ProcessPoolExecutor] = None
"""核心进程池"""

sub_process_pool: Optional[ProcessPoolExecutor] = None

thread_pool: Optional[ThreadPoolExecutor] = None
"""observer处理线程池"""

registered_image_converter: Dict[str, Callable] = {
    'raw_image': lambda x: x,
    'shm': lambda shm_name: load_image_from_shared_memory(shared_memory_name=shm_name),
}
"""已注册的图像转换器字典"""


def initial_handler_pool(max_worker: int = 8,
                         initial_callback: Callable = None,
                         initial_callback_arguments: tuple = None) -> Tuple[ProcessPoolExecutor, ProcessPoolExecutor]:
    """初始化进程

    .. Note::

        在主进程的某些变量可能不会继承到子进程，子进程中需要的函数必须在此方法中执行。

        如：数据库连接、特殊的变量、图像转换器字典、detect_handler等等


    Args:
        initial_callback (Callable, optional): 进程池子进程初始化回调参数. Defaults to None.
        initial_callback_arguments (tuple, optional): 回调函数参数. Defaults to None.
    """
    global process_pool, sub_process_pool

    initial_callback_arguments = initial_callback_arguments if initial_callback_arguments else tuple()

    if not initial_callback:
        initial_callback_wrapped = auto_detect_handler
        initial_callback_arguments = tuple()
    else:
        def initial_callback_wrapped(*args):
            auto_detect_handler()
            initial_callback(*args)

    process_pool = ProcessPoolExecutor(max_workers=max_worker, initializer=initial_callback_wrapped,
                                       initargs=initial_callback_arguments)
    sub_process_pool = ProcessPoolExecutor(max_workers=max_worker, initializer=initial_callback_wrapped,
                                           initargs=initial_callback_arguments)
    return process_pool, sub_process_pool


def initial_observer_pool() -> ThreadPoolExecutor:
    """初始化线程池"""
    global thread_pool
    thread_pool = ThreadPoolExecutor()

    return thread_pool


def sub_done_callback(sub_infer_task: Tuple[InferTask, dict], semaphore: Semaphore):
    def inner_done_callback(future: Future):
        if future._result:
            sub_infer_task[1].update(future._result)
        if exc := future.exception(0):
            sub_infer_task[1].update({str(sub_infer_task[0].handle_name): exc})
        semaphore.release()

    return inner_done_callback


def infer_callback(infer_task: InferTask) -> HandlerResult:
    """单次推理任务

    Note:: 运行流程

        1. 根据名字找到某个Handler - 找到某个模型
        2. 处理图片 - 找到图片转换器函数并且处理图片
        3. 推理/处理 - Handler + image + optional(kwargs) = result
        4.

    Args:
        infer_task (InferTask): 推理任务

    Returns:
        Any: 模型Handler处理的结果
    """
    image_converter_name = infer_task.image_converter_name if infer_task.image_converter_name else 'raw_image'

    # 根据名字找到某个Handler - 找到某个模型
    handle = get_handler(infer_task.handle_name)

    # 根据传入的名字获取加载图片的处理函数
    image_processor = registered_image_converter.get(image_converter_name)

    # 通过图片处理函数处理图片
    image = image_processor(infer_task.image_info)

    # 图片 + 模型处理 + （可选的参数） = 推理结果
    handler_result = handle.image_handle(image, **infer_task.parameter)

    # 二次识别
    sub_infer_tasks = []
    for sub_handler in [get_handler(sub_handler_name) for sub_handler_name in infer_task.sub_handlers]:
        sub_infer_tasks.extend(sub_handler.process_sub_task(image, handler_result))
    semaphore = Semaphore(0)

    for sub_infer_task in sub_infer_tasks:
        # FIXME: need another pool to execute sub_tasks.
        f = sub_process_pool.submit(infer_callback, sub_infer_task[0])
        # Done: use semaphore to avoid sub process error.
        f.add_done_callback(sub_done_callback(sub_infer_task, semaphore))

    # avoid to use block in result().
    # [future.result(1) for future in futures]

    [semaphore.acquire(timeout=1 / len(sub_infer_tasks)) for _ in range(len(sub_infer_tasks))]

    return handler_result


def _handler_process(handle_name: str,
                     image_info: Any = object(),
                     image_converter_name: str = 'raw_image',
                     other_kwargs: dict = None
                     ) -> Future:
    """系统进行推理的入口

    在主进程中调用，传入相关信息，子进程将会根据信息执行infer_callback并且返回一个Future。

    """
    return process_pool.submit(
        infer_callback,
        InferTask(
            handle_name=handle_name,
            image_info=image_info,
            image_converter_name=image_converter_name,
            parameter=other_kwargs if other_kwargs else {}
        )
    )


def handler_process(infer_task: InferTask):
    return process_pool.submit(
        infer_callback,
        infer_task,
    )


def observer_process(
        model_name: str,
        handler_result: HandlerResult
) -> List[Future]:
    futures = []
    for current_observer in filter(lambda x: model_name in x.required_models, _global_observer):
        futures.append(thread_pool.submit(current_observer.observer_judge_callback, model_name, handler_result))
    return futures
