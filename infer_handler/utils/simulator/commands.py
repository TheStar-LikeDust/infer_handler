# -*- coding: utf-8 -*-
"""命令集合

TODO:

    1. 检查命令：检查目前可用的Handlers和Observers
    2. 单张图片测试命令: 单独运行一次推理
    2.1 Handler测试
    2.2 Observer测试
    2.3 综合测试
    3. 生成报告：保存推理结果和判断结果 渲染Echarts
"""
import json
import os
import sys
import time
import uuid
from typing import Tuple

import click
import cv2
import numpy
import prettytable

from infer_handler.utils import observer_process, initial_observer_pool
from infer_handler.utils.detect import detect_handlers, detect_observer
from infer_handler.utils._global import _global_observer, _global_handlers, get_handler

from infer_handler.utils.simulator.engine import video_frame_cut, infer

# 主进程init
from infer_handler.triton_handler import set_client, InferenceServerClient

# TODO: 当前目录无法加入sys.path?
# TODO: 自定义参数可以选择导入组件的路径 所有命令

sys.path.append(os.getcwd())


@click.group()
def cli():
    pass


@click.command(name='show')
def show_components():
    """打印当前的组件: Handlers和Observers"""

    # TODO: 更多参数
    handlers_package = 'handlers'
    observers_package = 'observers'

    detect_handlers(handlers_package)
    detect_observer(observers_package)

    # build prettytable
    handlers_tb = prettytable.PrettyTable()
    handlers_tb.title = f'已加载的Infer Handlers'
    handlers_tb.field_names = ['index', 'InferHandler', 'package']
    handlers_tb._min_width = {'InferHandler': 20}

    [handlers_tb.add_row((row[0], row[1].name, row[1])) for row in enumerate(_global_handlers)]

    # output
    click.echo(f'* 当前的导入的包: {handlers_package}')
    click.echo(handlers_tb.get_string())

    # build prettytable
    observers_tb = prettytable.PrettyTable()
    observers_tb.title = f'已加载的Observers'
    observers_tb.field_names = ['index', 'Observer', 'instance']
    observers_tb._min_width = {'Observer': 20}

    [observers_tb.add_row((row[0], row[1].name, row[1])) for row in enumerate(_global_observer)]

    # output
    click.echo(f'* 当前的导入的包: {observers_package}')
    click.echo(observers_tb.get_string())


@click.command(name='check')
@click.option('handler', '--handler', default=None)
@click.option('observer', '--observer', default=None)
@click.option('image', '--image', default=None)
def check_component(handler=None, observer=None, image=None):
    """检查组件是否正常运行"""
    check_tb = prettytable.PrettyTable()
    check_tb.title = '可行性检查'
    check_tb.field_names = ['option name', 'option value']
    check_tb._min_width = {'option value': 20}

    check_tb.add_row(['handler', handler])
    check_tb.add_row(['observer', observer])
    check_tb.add_row(['image', image])

    click.echo(check_tb.get_string())

    # TODO: 更多参数
    handlers_package = 'handlers'
    observers_package = 'observers'

    detect_handlers(handlers_package)
    detect_observer(observers_package)

    if not image:
        test_image = numpy.ndarray((1080, 1920, 3), dtype=numpy.uint8)
        click.echo('* 未指定图片 - 生成空白图片')
    else:
        test_image = cv2.imread(image)
        click.echo(f'* 已指定图片 - {image}')

    if handler:
        handler_class = get_handler(handler_name=handler)
        click.echo(f'* 已加载Handler {handler_class}')

        test_infer_result = handler_class.image_handle(image=test_image)
        click.echo(f'* 推理完成 - result: {test_infer_result}')

        if observer_instance := [_ for _ in _global_observer if _.name == observer]:
            observer_instance = observer_instance[0]
            click.echo(f'* 已加载Observer {handler_class}')

            observer_instance.model_result_mapper[handler] = test_infer_result
            test_judge_result = observer_instance.judge()

            click.echo(f'* 判断完成 - result: {bool(test_judge_result)}')


@click.command('infer')
@click.option('video', '--video', prompt=True)
@click.option('handler', '--handler', multiple=True, prompt=True)
@click.option('triton', '--triton', default='', type=click.STRING)
def do_infer(video: str, handler: Tuple[str], triton: str):
    """生成检测结果"""

    # 建立Triton连接
    if triton:
        set_client(InferenceServerClient(url=triton))

    handlers_package = 'handlers'

    detect_handlers(handlers_package)

    click.echo('-' * 80)

    for handler_name in handler:
        get_handler(handler_name)
        click.echo(f'* Load Handler: {handler_name}')

    click.echo('-' * 80)
    # 截帧

    image_list = video_frame_cut(video)
    click.echo(f'* 截帧完成')
    click.echo(f'* 图片共计: {len(image_list)}张')

    click.echo('-' * 80)
    # 模型推理
    infer_result = infer(image_list, *handler)

    click.echo('-' * 80)
    file_name = f'infer_result_{uuid.uuid1().node}.json'

    click.echo(f'* 保存推理结果至 -> {file_name}')
    with open(file_name, 'w') as f:
        json.dump(infer_result, f)
    click.echo(f'* 成功保存至 -> {file_name}')


@click.command(name='judge')
@click.option('result', '--result')
def do_judge(result):
    """逻辑判断"""

    observers_package = 'observers'
    detect_observer(observers_package)
    initial_observer_pool()

    click.echo('-' * 80)
    click.echo('* 读取推理结果中...')
    with open(result) as f:
        infer_result = json.load(f)
        click.echo(f'* 读取完成 - <{result}>')

    click.echo('-' * 80)
    for frame_result in infer_result:
        for model_name in frame_result:
            observer_process(model_name, frame_result[model_name])

    click.echo('* 触发事件的Observer:')
    [click.echo(f'* Observer: {ob.name} 已触发') for ob in _global_observer if ob.last_trigger_time > 0]

    click.echo('-' * 80)


cli.add_command(show_components)
cli.add_command(check_component)
cli.add_command(do_infer)
cli.add_command(do_judge)

if __name__ == '__main__':
    cli()
