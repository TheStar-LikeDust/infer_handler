# -*- coding: utf-8 -*-
"""核心类

所有的InferHandler都需要继承自InferHandler类

"""
from abc import abstractmethod
from logging import getLogger
from typing import Any, Optional

logger = getLogger('infer_handler')


class InferHandlerMeta(type):
    """ImageHandler元类"""

    def __new__(mcs, name, bases, attrs):
        if not attrs.get('name'):
            attrs['name'] = name
        return super().__new__(mcs, name, bases, attrs)


class InferHandler(object, metaclass=InferHandlerMeta):
    """ImageHandler基类

    提供了若干个InferHandler通用方法。
    """
    name: str
    """类名，也可以自定义"""

    keep_context: bool = False
    """是否保存image_handle时的上下文(前处理等信息)"""

    def __new__(cls, *args, **kwargs):
        """屏蔽实例化方法"""
        return cls

    @classmethod
    def initial_handler(cls):
        """初始化模板方法 - 不能重写"""
        try:
            cls._initial_handler()
        except Exception as e:
            logger.error(f'Handler: {cls.name} initial failed.', exc_info=e)
            return False
        else:
            return True

    @classmethod
    @abstractmethod
    def _initial_handler(cls):
        """加载一些当前Handler需要的资源 - 由子类重写"""
        pass

    @classmethod
    @abstractmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        """前处理抽象方法 - 由子类重写"""
        pass

    @classmethod
    @abstractmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        """推理抽象方法 - 由子类重写"""
        pass

    @classmethod
    @abstractmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        """后处理抽象方法 - 由子类重写"""
        pass

    @classmethod
    def image_handle(cls, image: Any, **kwargs) -> dict:
        """通用的模板方法  - 不能重写"""

        pre_result = cls._pre_process(image, **kwargs)
        kwargs.update({'pre_result': pre_result})

        infer_result = cls._infer_process(image, **kwargs)
        kwargs.update({'infer_result': infer_result})

        post_result = cls._post_process(image, **kwargs)

        if cls.keep_context:
            return {'post_result': post_result, **kwargs}
        else:
            return post_result
