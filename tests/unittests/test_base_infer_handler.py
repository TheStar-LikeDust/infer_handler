import sys
import unittest

from infer_handler import InferHandler

class Custom(InferHandler):
    pass

class MyTestCase(unittest.TestCase):
    def test_methods_image_handle(self):
        """template"""

        InferHandler.image_handle(object())

    def test_attribute_name(self):

        assert Custom.name == 'Custom'


if __name__ == '__main__':
    unittest.main()
