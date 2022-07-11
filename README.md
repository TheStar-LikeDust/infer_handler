### Readme

> 模型包装器，提供了一套通用的API模板和相应的工具，用于模型的并行推理和测试。

### 使用

1. 编写自己的Handler

```python
from typing import Any, Optional

from infer_handler import InferHandler


class BlankHandler(InferHandler):
    @classmethod
    def _pre_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return image.shape

    @classmethod
    def _infer_process(cls, image: Any, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    def _post_process(cls, image: Any, **kwargs) -> Optional[Any]:
        return {
            'shape': kwargs.get('pre_result'),
            'info': 'info_content'
        }

```

2. 初始化进程池

```python

from infer_handler.utils.detect import detect_handlers
from infer_handler.utils.worker import initial_handler_pool

# 自动检测package路径下文件
detect_handlers('tests.asset.mock_handlers')

# 初始化进程池
initial_handler_pool()
``` 

3. 调用 - 并行处理

```python
from infer_handler.utils.worker import handler_process

# 指定Handler 放入图片 设置其他参数
future = handler_process('blank_handler', image_info=image, other_kwargs={'info': 'info_content'})

# 得到结果
result = future.result()
```

## roadmap and TODO List:

- unitest in detect\observer\worker
- doc and tutorial


