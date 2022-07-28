# -*- coding: utf-8 -*-
"""Observer - 判断逻辑

一个简单的逻辑判断类，用于处理连续的视频帧处理系统中复杂的逻辑判断。
"""
import time
from abc import abstractmethod
from logging import getLogger
from typing import Any, List, Tuple, NoReturn, Dict, Optional

from collections import deque


# TODO: type alies

# List[Tuple[bool, Any]]

class ObserverMeta(type):
    def __new__(mcs, name, bases, attrs):
        attrs['name'] = name
        return super().__new__(mcs, name, bases, attrs)


class Observer(object, metaclass=ObserverMeta):
    """逻辑判断类基类

    子类继承并实现不同的judge方法，将每一帧判断结果放入缓存队列中，如果报警比例达到触发比例，则执行触发回调。
    """
    name: str
    """类名"""

    required_labels = []
    """需要的模型"""

    judge_result_queue = List[Tuple[bool, Any]]
    """判断结果和识别结果缓存队列 - 滑动窗口"""

    trigger_rate: float = 0.5
    """滑动窗口中的达到触发预警的比例"""

    trigger_cache_length: int = 10
    """滑动窗口长度"""

    last_trigger_time: float = 0.0
    """上次触发报警的时间"""

    trigger_time_gap: int = 60
    """两次触发的时间间隔"""

    model_result_mapper = Dict[str, Any]
    """所需要的模型结果字典"""

    trigger_limit: int
    """触发帧数"""

    def __init__(self, logger=None):
        """初始化

        Args:
            cache_length (int, optional): 基于双端队列的缓存的滑动窗口长度. Defaults to 10.
            trigger_rate (float, optional): 触发判断比例. Defaults to 0.5.
            logger (_type_, optional): 单独的日志类. Defaults to None.
        """
        self.judge_result_queue = deque(maxlen=self.trigger_cache_length)

        self.trigger_limit = int(self.trigger_rate * self.judge_result_queue.maxlen)

        # self.trigger_rate = trigger_rate

        self.logger = logger if logger else getLogger(self.name)

        self.model_result_mapper = {_: None for _ in self.required_labels}

        # 填充
        [self.judge_result_queue.append((False, None)) for _ in range(self.trigger_cache_length)]

    @abstractmethod
    def judge(self) -> Optional[bool]:
        """每一帧的具体判断逻辑"""
        pass

    @abstractmethod
    def trigger(self) -> NoReturn:
        """触发的事件回调"""
        pass

    def observer_judge_callback(self, model_name: str, result: Any):
        """模型处理判断函数

        Args:
            model_name (str): 模型名字
            result (Any): 模型推理结果
        """

        # update
        self.model_result_mapper[model_name] = result

        # case: 判断是否满足一次judge的条件
        if set(self.model_result_mapper.keys()) == set(self.required_labels):
            try:
                judge_result = self.judge()
            except Exception as e:
                self.logger.info(f'{self.name} judge error.', exc_info=e)
                judge_result = False

            # 先清空
            self.clear_model_result()
            # 添加结果
            self.judge_result_queue.append((judge_result, result))
            # 判断是否触发
            self.check_trigger()

    def clear_model_result(self):
        self.model_result_mapper.clear()

    def check_trigger(self):
        trigger_frame_flag = len(list(filter(lambda x: x[0], self.judge_result_queue))) >= self.trigger_limit

        trigger_time_flag = time.time() - self.last_trigger_time > self.trigger_time_gap

        # case: 大于阈值 且 上次触发间隔大于设置的间隔
        if trigger_frame_flag and trigger_time_flag:
            self.last_trigger_time = time.time()
            self.trigger()
