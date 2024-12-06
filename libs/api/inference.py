# modified based on:
# https://github.com/open-mmlab/mmdetection/blob/v2.28.0/mmdet/apis/inference.py
# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import torch
from mmcv.parallel import collate, scatter

from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp


def inference_one_image(model, img):
    """Inference on an image with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img (np.ndarray): Image data.
    Returns:
        img (np.ndarray): Image data with shape (width, height, channel).
        preds (List[np.ndarray]): Detected lanes.
    """
    # 입력 이미지를 CLRerNet이 예상하는 크기로 리사이즈
    img = cv2.resize(img, (1640, 590))
    ori_shape = img.shape
    
    data = dict(
        img=img,
        img_shape=ori_shape,
        ori_shape=ori_shape,
        filename=None,
        sub_img_name=None,
        gt_points=[],  # 빈 리스트로 초기화
        id_classes=[],  # 빈 리스트로 초기화
        id_instances=[],  # 빈 리스트로 초기화
    )

    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False
    device = next(model.parameters()).device  # model device

    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    lanes = results[0]['result']['lanes']
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])

    return img, preds


def get_prediction(lanes, ori_h, ori_w):
    preds = []
    for lane in lanes:
        lane = lane.cpu().numpy()
        xs = lane[:, 0]
        ys = lane[:, 1]
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * ori_h
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
        interp_pred = interp(pred, n=5)
        preds.append(interp_pred)
    return preds
