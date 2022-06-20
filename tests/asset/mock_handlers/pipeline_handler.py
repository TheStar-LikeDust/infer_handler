# -*- coding: utf-8 -*-
"""Pipeline 处理器

"""
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


class PipelineHandler(TritonHandler):
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
