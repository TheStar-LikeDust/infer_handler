# -*- coding: utf-8 -*-
"""测试

两个Pool互相调用，是否会引发程序无法退出的问题。

"""

from concurrent.futures import ProcessPoolExecutor

main_pool = None

sub_pool = None


def init_pool():
    global main_pool, sub_pool

    main_pool = ProcessPoolExecutor(max_workers=2)
    sub_pool = ProcessPoolExecutor(max_workers=2)


def sub_simple_task():
    pass


def simple_task():
    sub_pool.submit(sub_simple_task)


if __name__ == '__main__':
    init_pool()

    main_pool: ProcessPoolExecutor
    sub_pool: ProcessPoolExecutor

    f = main_pool.submit(simple_task)

    exit(0)
    f.result()
    exit(0)

