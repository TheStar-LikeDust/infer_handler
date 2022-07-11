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
from typing import Tuple
from types import FunctionType

import click
import cv2
import numpy
import prettytable

from infer_handler.utils import observer_process, initial_observer_pool
from infer_handler.utils.detect import detect_handlers, detect_observer, auto_detect_handler, auto_detect_observer
from infer_handler.utils._global import _global_observer, _global_handlers, get_handler

from infer_handler.utils.simulator.engine import video_frame_cut, infer

# 主进程init

# TODO: 当前目录无法加入sys.path?
# TODO: 自定义参数可以选择导入组件的路径 所有命令

sys.path.append(os.getcwd())


@click.group()
def cli():
    pass


common_options = {
    'http_client':
        click.Option(
            param_decls=['--http_client', 'http_client'],
            is_flag=True,
            help='[Common] 是否使用HTTP Client',
        ),

    'triton_connect':
        click.Option(
            param_decls=['--triton_connect', 'triton_connect'],
            default='',
            help='[Common] Triton连接地址',
        ),
    'handlers':
        click.Option(
            param_decls=['--handlers', 'handlers'],
            default='handlers',
            help='[Common] Handlers 包的路径',
        ),
    'observers':
        click.Option(
            param_decls=['--observers', 'observers'],
            default='observers',
            help='[Common] Observer 包的路径',
        ),
}
"""公用的Click Option"""


def base_command(name, options):
    """Add common options."""

    def base(wrap_function: FunctionType):
        wrap_command = click.Command(
            name=name if name else wrap_function.__name__,
            callback=wrap_function,
        )

        if options:
            for option_name in options:
                wrap_command.params.append(common_options[option_name])

        return wrap_command

    return base


def process_base_kwargs(kwargs_dict):
    """Process common options"""

    # 开启HTTP
    from infer_handler import switch_to_http
    if http_client := kwargs_dict.get('http_client'):
        switch_to_http()

    # 连接Triton
    from infer_handler.triton_handler import set_client, InferenceServerClient
    if triton_url := kwargs_dict.get('triton_connect'):
        set_client(client=InferenceServerClient(url=triton_url))

    # 导入组件
    if observer_package := kwargs_dict.get('observers'):
        auto_detect_observer(observer_package)

    if handler_package := kwargs_dict.get('handlers'):
        auto_detect_handler(handler_package)


@base_command(name='show', options=['handlers', 'observers'])
def show_components(**kwargs):
    """打印当前的组件: Handlers和Observers"""

    process_base_kwargs(kwargs)

    # build prettytable
    handlers_tb = prettytable.PrettyTable()
    handlers_tb.title = f'已加载的Infer Handlers'
    handlers_tb.field_names = ['index', 'InferHandler', 'package']
    handlers_tb._min_width = {'InferHandler': 20}

    [handlers_tb.add_row((row[0], row[1].name, row[1])) for row in enumerate(_global_handlers)]

    # output
    click.echo(f'* 当前的导入的包路径: {os.path.abspath(kwargs.get("handlers"))}')
    click.echo(handlers_tb.get_string())

    # build prettytable
    observers_tb = prettytable.PrettyTable()
    observers_tb.title = f'已加载的Observers'
    observers_tb.field_names = ['index', 'Observer', 'instance']
    observers_tb._min_width = {'Observer': 20}

    [observers_tb.add_row((row[0], row[1].name, row[1])) for row in enumerate(_global_observer)]

    # output
    click.echo(f'* 当前的导入的包路径: {os.path.abspath(kwargs.get("observers"))}')
    click.echo(observers_tb.get_string())


@click.option('observer', '--observer', default=None, help='指定测试的observer名')
@click.option('handler', '--handler', default=None, help='指定测试的handler名')
@click.option('image', '--image', default=None, help='测试图片')
@base_command(name='check', options=['handlers', 'observers'])
def check_component(handler=None, observer=None, image=None, **kwargs):
    """检查组件是否正常运行"""
    process_base_kwargs(kwargs)

    check_tb = prettytable.PrettyTable()
    check_tb.title = '可行性检查'
    check_tb.field_names = ['option name', 'option value']
    check_tb._min_width = {'option value': 20}

    check_tb.add_row(['handler', handler])
    check_tb.add_row(['observer', observer])
    check_tb.add_row(['image', image])

    click.echo(check_tb.get_string())

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


@click.option('video', '--video', prompt=True)
@click.option('handler', '--handler', multiple=True, prompt=True)
@base_command(name='infer', options=['handlers', 'http_client', 'triton_connect'])
def do_infer(video: str, handler: Tuple[str], **kwargs):
    """生成检测结果"""
    process_base_kwargs(kwargs)

    click.echo('-' * 80)

    for handler_name in handler:
        get_handler(handler_name)
        click.echo(f'* Success load Handler: {handler_name}')

    click.echo('-' * 80)
    # 截帧

    image_list = video_frame_cut(video)
    click.echo(f'* 截帧完成')
    click.echo(f'* 图片共计: {len(image_list)}张')

    click.echo('-' * 80)
    # 模型推理
    infer_result = infer(image_list, *handler)

    click.echo('-' * 80)
    file_name = f'infer_result_{int(time.time())}.json'

    click.echo(f'* 保存推理结果至 -> {file_name}')
    with open(file_name, 'w') as f:
        json.dump(infer_result, f)
    click.echo(f'* 成功保存至 -> {file_name}')


@click.option('result', '--result')
@base_command(name='judge', options=['observers'])
def do_judge(result, **kwargs):
    """逻辑判断"""

    process_base_kwargs(kwargs)

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
