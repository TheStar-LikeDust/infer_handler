# -*- coding: utf-8 -*-
"""setup with setuptools."""

from setuptools import setup, find_packages
from infer_handler import __version__

setup(
    name='infer_handler',
    version=__version__,
    description='A model handler template and toolkit make it easy to test/infer/deploy.',
    author='Logic',
    author_email='logic.irl@outlook.com',
    url='https://github.com/TheStar-LikeDust/infer_handler',
    python_requires='>=3.8',
    packages=find_packages(exclude=['tests*']),
    # patched
    install_requires=[
        'Click',
        'tqdm',
        'prettytable',
    ],
    entry_points='''
        [console_scripts]
        infer_handler_tool=infer_handler.utils.simulator.commands:cli
    ''',
    license='Apache License 2.0'
)
