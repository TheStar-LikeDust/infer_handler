import unittest
from typing import Any, Optional

from infer_handler import InferHandler


class Blank(InferHandler):
    pass


class Custom(InferHandler):

    @classmethod
    def _initial_handler(cls):
        cls.initial_flag = True

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return 'pre_result'

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return 256

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return kwargs.get('infer_result') ** 2


class InferHandlerTestCase(unittest.TestCase):
    def test_methods_image_handle(self):
        """Blank image_handler method"""
        Blank.image_handle(object())

    def test_property(self):
        """Property"""

        with self.subTest('name: (Fixed)name of class'):
            assert Blank.name == 'Blank' == Blank.__name__

        with self.subTest('keep_context: default False'):
            assert Blank.keep_context == False

    def test_template_methods(self):
        """Common template methods."""
        Custom.keep_context = False

        with self.subTest('initial_handler'):
            assert not hasattr(Custom, 'initial_flag')
            Custom.initial_handler()
            assert Custom.initial_flag == True

        with self.subTest('image_handle'):
            image_handler_result = Custom.image_handle(object(), **{'kw': 'kw_content'})

            assert image_handler_result == 65536

    def test_keep_context_in_handling(self):
        """Keep_context will save pre/infer/post_result into result."""

        Custom.keep_context = True

        image_handler_result = Custom.image_handle(object(), **{'kw': 'kw_content'})

        assert image_handler_result.get('pre_result') == 'pre_result'
        assert image_handler_result.get('infer_result') == 256
        assert image_handler_result.get('post_result') == 65536


if __name__ == '__main__':
    unittest.main()
