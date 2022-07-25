# -*- coding: utf-8 -*-
"""Manager 管理


"""
import signal

from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

pool_mapper = None


def init_pool():
    global main_pool, sub_pool, pool_mapper

    pool_mapper = {}
    sub_pool = ProcessPoolExecutor(max_workers=2)
    main_pool = ProcessPoolExecutor(max_workers=2)

    pool_mapper['main'] = main_pool
    pool_mapper['sub'] = sub_pool


def sub_simple_task():
    pass


def simple_task():
    print('sub!')
    pool_mapper['sub'].submit(sub_simple_task).result()
    print('sub done!')


if __name__ == '__main__':
    init_pool()

    p = pool_mapper['main']

    f = p.submit(simple_task)
    f.result()

    signal.alarm(3)
