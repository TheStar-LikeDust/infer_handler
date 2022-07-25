# -*- coding: utf-8 -*-
"""性能测试包


"""
import time

import cv2
from tritonclient.grpc import InferenceServerClient

image = cv2.imread(r'tests/mocks/p0.jpg')
_client = InferenceServerClient(url='localhost:8352')

from tritonclient.grpc import InferInput, InferRequestedOutput

input_image = InferInput('images', [1, 3, 960, 960], "FP32")

output = InferRequestedOutput('output')
output_416 = InferRequestedOutput('416')
output_402 = InferRequestedOutput('402')

triton_inputs = [input_image]
triton_outputs = [output, output_416, output_402]


def callback(result, error):
    pass


def loop_infer_200():
    # pass
    global image

    from tqdm import trange

    for _ in trange(200):
        s = time.time()
        pre_result = pre_process(image)
        input_image.set_data_from_numpy(pre_result)

        print('pre cost', time.time() - s)
        # TODO: 异步
        # infer_result = client.async_infer(
        #     model_name='fire_smog_yolov5_v3',
        #     inputs=triton_inputs,
        #     outputs=triton_outputs,
        #     callback=callback,
        #     client_timeout=1,
        # )
        s = time.time()
        infer_result = _client.infer(
            model_name='fire_smog_yolov5_v3',
            inputs=triton_inputs,
            outputs=triton_outputs,
            client_timeout=1,
        )
        print('infer cost', time.time() - s)

        s = time.time()
        res = post_process(infer_result, image.shape[:2])
        print('post cost', time.time() - s)


import cv2
import numpy as np
import torch
import torchvision


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
              stride=32):
    """图片归一化"""
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


def nms(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    xc = prediction[..., 4] > conf_thres  # candidates
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]

    return output


def clip_coords(boxes, img_shape):
    """查看是否越界"""
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
        坐标对应到原始图像上，反操作：减去pad，除以最小缩放比例
        :param img1_shape: 输入尺寸
        :param coords: 输入坐标
        :param img0_shape: 映射的尺寸
        :param ratio_pad:
        :return:
    """

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
    coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
    coords[:, :4] /= gain  # 将box坐标对应到原始图像上
    clip_coords(coords, img0_shape)  # 边界检查
    return coords


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pre_process(img):
    """执行前向操作预测输出"""
    # 图片填充并归一化
    img_size = (960, 960)  # 图片缩放大小
    img = letterbox(img, img_size, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # 归一化
    img = img.astype(dtype=np.float32)
    img /= 255.0

    # 维度扩张
    img = np.expand_dims(img, axis=0)

    return img


def post_process(infer_result, src_size):
    conf_thres = 0.25  # 置信度阈值
    iou_thres = 0.45  # iou阈值
    class_num = 2  # 类别数
    img_size = (960, 960)  # 图片缩放大小
    stride = [8, 16, 32]
    names = ['Fire', 'Smog']
    anchor_list = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor = np.array(anchor_list).astype(np.float32).reshape(3, -1, 2)

    area = img_size[0] * img_size[1]
    size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]
    feature = [[int(j / stride[i]) for j in img_size] for i in range(3)]

    infer_result = [infer_result.as_numpy('output'), infer_result.as_numpy('416'), infer_result.as_numpy('402')]

    pred0 = infer_result[0]
    pred1 = infer_result[1]
    pred2 = infer_result[2]

    # 提取出特征
    y = []
    y.append(torch.tensor(pred0.reshape(-1, size[0] * 3, 5 + class_num)).sigmoid())
    y.append(torch.tensor(pred2.reshape(-1, size[1] * 3, 5 + class_num)).sigmoid())
    y.append(torch.tensor(pred1.reshape(-1, size[2] * 3, 5 + class_num)).sigmoid())

    grid = []
    for k, f in enumerate(feature):
        grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

    z = []
    for i in range(3):
        src = y[i]

        xy = src[..., 0:2] * 2. - 0.5
        wh = (src[..., 2:4] * 2) ** 2
        dst_xy = []
        dst_wh = []
        for j in range(3):
            dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + torch.tensor(grid[i])) * stride[i])
            dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
        src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
        src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
        z.append(src.view(1, -1, 5 + class_num))

    results = torch.cat(z, 1)
    results = nms(results, conf_thres, iou_thres)

    # 映射到原始图像
    img_shape = (960, 960)
    res = {k: [] for k in names}
    for det in results:  # detections per image
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_shape, det[:, :4], src_size).round()
            for *xyxy, conf, cls in det:
                label = int(cls)  # 类别
                # 置信度
                confidence = round(float(conf), 2)
                x1 = max(int(xyxy[0]), 0)
                y1 = max(int(xyxy[1]), 0)
                x2 = max(int(xyxy[2]), 0)
                y2 = max(int(xyxy[3]), 0)
                box = [x1, y1, x2, y2, label, confidence]
                if cls == 0:
                    res[names[0]].append(box)
                elif cls == 1:
                    res[names[1]].append(box)

    return res


if __name__ == '__main__':
    loop_infer_200()
