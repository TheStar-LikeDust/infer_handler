# -*- coding: utf-8 -*-
"""核心类

"""
from abc import abstractmethod
from logging import getLogger
from typing import Any, Optional

logger = getLogger('infer_handler')


class InferHandlerMeta(type):
    """ImageHandler元类，初始化一些常用方法"""

    def __new__(mcs, name, bases, attrs):
        attrs['name'] = name
        module = attrs['__module__']
        if '.' in module:
            attrs['module_name'] = attrs['__module__'].split('.')[-1]
        else:
            attrs['module_name'] = module

        return super().__new__(mcs, name, bases, attrs)


class InferHandler(object, metaclass=InferHandlerMeta):
    """ImageHandler基类"""
    module_name: str
    """模块名"""
    name: str
    """类名"""

    keep_context: bool = False
    """保存上下文"""

    # block instance
    def __new__(cls, *args, **kwargs):
        return InferHandler

    @classmethod
    def initial_handler(cls):
        """初始化模板方法"""
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
        pass

    @classmethod
    @abstractmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        """前处理抽象方法"""
        pass

    @classmethod
    @abstractmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        """推理抽象方法"""
        pass

    @classmethod
    @abstractmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        """后处理抽象方法"""
        pass

    @classmethod
    def image_handle(cls, image: Any, **kwargs) -> dict:
        """通用的模板方法"""

        pre_result = cls._pre_process(image, **kwargs)
        kwargs.update({'pre_result': pre_result})

        infer_result = cls._infer_process(image, **kwargs)
        kwargs.update({'infer_result': infer_result})

        post_result = cls._post_process(image, **kwargs)

        if cls.keep_context:
            return {'post_result': post_result, **kwargs}
        else:
            return post_result
