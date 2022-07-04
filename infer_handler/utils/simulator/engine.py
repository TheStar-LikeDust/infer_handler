# -*- coding: utf-8 -*-
"""检查核心引擎

负责启动一个简单的多进程推理服务

"""
from multiprocessing import Barrier
import threading
import time

from infer_handler.utils import initial_handler_pool, initial_observer_pool, handler_process, observer_process

client_url: str = ''
"""Triton地址"""

process_number: int = 8
"进程个数"

barrier = Barrier(9)

barrier_timeout = 10


def set_triton_client(url: str):
    """设置url"""
    global client_url
    client_url = url


def set_process_number(number: int):
    """设置进程个数"""
    global process_number, barrier
    process_number = number
    barrier = Barrier(number + 1)


def _engine_callback():
    from infer_handler.triton_handler import set_client
    from tritonclient.http import InferenceServerClient
    set_client(InferenceServerClient(url=client_url))
    print('Triton server连接成功')

    from infer_handler.utils.worker import registered_image_converter
    from shared_memory_toolkit import load_image_from_shared_memory

    registered_image_converter['shm'] = lambda x: load_image_from_shared_memory(x)
    print('共享内存转换器注册成功')

    barrier.wait()


def _start_engine():
    initial_observer_pool()

    process_pool = initial_handler_pool(max_worker=process_number, initial_callback=_engine_callback)
    [process_pool.submit(lambda: None) for _ in range(process_number)]

    barrier.wait()
    print('模拟引擎启动完成')


def infer_image(image_sequence, handler_name, observer_name: str = None):
    from infer_handler.utils import worker

    if not (worker.process_pool and worker.thread_pool):
        _start_engine()

    image_number = len(image_sequence)
    print('图片数量', image_number)

    # 建立推理结果列表
    infer_result = []

    # 多线程模拟多路识别
    from shared_memory_toolkit import dump_image_into_shared_memory
    def consuming():
        while image_sequence:
            # 获取图片 + 共享内存名（线程名字）-> 写入共享内存
            image = image_sequence.pop()
            shm_name = threading.current_thread().getName()
            dump_image_into_shared_memory(shm_name, image)
            # 放入进程池推理
            f = handler_process(handler_name, shm_name, 'shm')
            # 完成推理任务后放入infer_result中

            # 打印
            # f.add_done_callback(lambda x: infer_result.append(x.result()) or print(x.result()))
            # 不打印
            f.add_done_callback(lambda x: infer_result.append(x.result()))

    # 创建线程
    # 线程名自然互斥 - 保持共享内存名唯一
    # 设置守护线程 - 保证正常退出
    thread_number = process_number
    threads = [threading.Thread(target=consuming) for _ in range(thread_number)]
    [t.setDaemon(True) for t in threads]
    [t.start() for t in threads]

    from tqdm import tqdm
    with tqdm(total=image_number) as bar:

        while len(infer_result) < image_number:
            bar.n = 0
            bar.update(len(infer_result))
            time.sleep(0.1)

    print('模型处理完毕')

    if observer_name:
        for res in infer_result:
            observer_process(observer_name, res)

    exit(0)
