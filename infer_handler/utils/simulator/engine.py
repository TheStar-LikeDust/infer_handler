# -*- coding: utf-8 -*-
"""检查核心引擎

负责启动一个简单的多进程推理服务

"""
from multiprocessing import Barrier
import threading
import time
from typing import List

import numpy
import tqdm
import cv2

from infer_handler.utils import initial_handler_pool, initial_observer_pool, handler_process, observer_process
from infer_handler.utils._global import get_handler

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


def video_frame_cut(video_path: str):
    """截帧
    TODO: 目前固定1秒十帧 往上取整
    """
    v = cv2.VideoCapture(video_path)

    gap_frame = int(25 / 10)
    total_frame = int(v.get(7) / gap_frame)
    count = 0

    image_list = []

    with tqdm.tqdm(total=total_frame) as bar:
        bar.set_description_str('CV2截帧')
        while True:
            _, image = v.read()
            if not _:
                break

            if count % 2 == 0:
                image_list.append(image)
                bar.update()

            count += 1

    return image_list


def infer(image_list=List[numpy.ndarray], *handlers: str):
    """主进程推理"""

    image_number = len(image_list)
    infer_result = [{} for _ in range(image_number)]

    for handler_name in handlers:
        handler = get_handler(handler_name)

        print(f'* 正在使用 Handler: {handler_name}')

        with tqdm.tqdm(total=image_number) as bar:
            for index in range(image_number):
                infer_result[index][handler_name] = handler.image_handle(image=image_list[index])
                bar.update()

    return infer_result
