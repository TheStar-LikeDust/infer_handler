import unittest

from . import loop_infer_200


class MyTestCase(unittest.TestCase):
    def test_something(self):
        loop_infer_200()


if __name__ == '__main__':
    unittest.main()
