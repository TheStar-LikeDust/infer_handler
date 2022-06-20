## 简单使用教程

### 0. 下载&安装

克隆库
> git clone https://github.com/TheStar-LikeDust/infer_handler.git

安装
> python setup.py install

### 1. 最简单使用使用

- 新建Python包: 名字为handler

```
handlers
    __init__.py
```

- 在路径下创建py模块

```
handlers
    __init__.py
    alpha_handler.py
```

- 编辑alpha_handler模块

```python

from typing import Any, Optional

from infer_handler import InferHandler


class AlphaHandler(InferHandler):
    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return {
            'info': 'alpha',
        }
```

- 调用

```python
# 主路径下

# 构建子进程初始化函数
import numpy


def initial_callback():
    from infer_handler.utils import detect_handlers
    detect_handlers('handlers')


from infer_handler.utils import initial_pool, handler_process

if __name__ == '__main__':
    # 初始化子进程池
    initial_pool(initial_callback=initial_callback)

    # handler_name: 类名
    handler_name = 'AlphaHandler'
    # image_info: 图片信息
    image_info = numpy.ndarray((1080, 1920, 3), dtype=numpy.uint8)

    future = handler_process(
        handler_name,
        image_info,
    )

    print(future.result())
```

### 2. 手动加载网络的InferHandler - Yolov5

需要单独设置的Handler，需要继承_initial_handler，子进程会自动执行此初始化方法。

> 推荐将一些常量设为模块变量，特殊设置为类变量

```python
from typing import Any, Optional
from infer_handler import InferHandler

import torch


class Beta(InferHandler):

    @classmethod
    def _initial_handler(cls):
        cls.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return cls.model(image)

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        # 将数据转换为可序列化的值
        res = [_.numpy() for _ in kwargs.get('infer_result').pred]

        return res
```

### 3. 基于Triton的InferHandler

> 2022.6.20更新

##### 在子进程初始化函数中手动设置Triton

```python
def initial_callback():
    from infer_handler.utils import detect_handlers
    detect_handlers('handlers')

    from infer_handler.triton_handler import set_client
    from tritonclient.http import InferenceServerClient

    # set_client传入一个Triton Client对象
    set_client(InferenceServerClient(url=r'192.168.1.110:8000'))
```

##### 继承infer_handler.triton_handler的Handler

```python
from typing import Any, Optional

from numpy import ndarray, expand_dims, int32, float32, array

from infer_handler.triton_handler import TritonHandler, InferInput, InferRequestedOutput

input_image = InferInput('image', [1, 1080, 1920, 3], "UINT8")
input_infer_size = InferInput('infer_size', [1, 2], "INT32")
input_nms_conf_thresholds = InferInput('nms_conf_thresholds', [1, 2], "FP32")

input_infer_size.set_data_from_numpy(array([(1280, 1280)], dtype=int32))
input_nms_conf_thresholds.set_data_from_numpy(array([(0.4, 0.25)], dtype=float32))

output = InferRequestedOutput('result')
model_name = 'pedestrian_yolov5_pipeline'


class Gamma(TritonHandler):
    """Demo"""

    triton_model_name = 'pedestrian_yolov5_pipeline'
    triton_inputs = [input_image, input_nms_conf_thresholds, input_infer_size]
    triton_outputs = [output]

    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        cls.triton_inputs[0].set_data_from_numpy(expand_dims(image, axis=0))

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return kwargs.get('infer_result').as_numpy('result')

```