# -*- coding: utf-8 -*-
"""Observer - 判断逻辑

一个简单的逻辑判断类，用于处理连续的视频帧处理系统中复杂的逻辑判断。
"""
from abc import abstractmethod
from logging import getLogger
from typing import Any, List, Tuple, NoReturn

from collections import deque


class Observer(object):
    """逻辑判断类基类

    子类继承并实现不同的judge方法，将每一帧判断结果放入缓存队列中，如果报警比例达到触发比例，则执行触发回调。
    """
    name = 'observer'
    """判断类类名，也是日志类的名字"""

    filter_code = ['001', 'person_car', 'smoking']
    """过滤码列表"""

    judge_result_queue = List[Tuple[bool, Any]]
    """判断结果和识别结果缓存队列"""

    trigger_rate: float = 0.5
    """触发报警的比例"""

    def __init__(self, cache_length: int = 10, trigger_rate: float = 0.5, logger=None):
        """初始化

        Args:
            cache_length (int, optional): 基于双端队列的缓存的滑动窗口长度. Defaults to 10.
            trigger_rate (float, optional): 触发判断比例. Defaults to 0.5.
            logger (_type_, optional): 单独的日志类. Defaults to None.
        """
        self.judge_result_queue = deque(maxlen=cache_length)
        self.trigger_rate = trigger_rate

        self.logger = logger if logger else getLogger(self.name)

        # 填充
        [self.judge_result_queue.append((False, None)) for _ in range(cache_length)]

    @abstractmethod
    def judge(self, result: Any) -> bool:
        """每一帧的具体判断逻辑"""
        pass

    @abstractmethod
    def trigger(self) -> NoReturn:
        """触发的事件回调"""
        pass


def observer_judge_callback(observer: Observer, result: Any):
    """模型处理判断函数

    Args:
        observer (Observer): 逻辑处理类
        result (Any): 推理结果
    """
    try:
        judge_result = observer.judge(result)
    except Exception as e:
        observer.logger.info(f'{observer.name} judge error.', exc_info=e)
        judge_result = False

    observer.judge_result_queue.append((judge_result, result))

    limit = observer.trigger_rate * observer.judge_result_queue.maxlen

    if len(list(filter(lambda x: x[0], observer.judge_result_queue))) >= limit:
        observer.trigger()
