import os

import cv2
import numpy as np
import paddle
import skimage


@paddle.no_grad()
def evaluate_error(gt_depth, pred_depth):
    # for numerical stability
    depth_mask = gt_depth > 0.0001
    error = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    _pred_depth = pred_depth[depth_mask]
    _gt_depth = gt_depth[depth_mask]
    n_valid_element = _gt_depth.shape[0]
    if n_valid_element > 0:
        diff_mat = paddle.abs(_gt_depth - _pred_depth)
        rel_mat = paddle.divide(diff_mat, _gt_depth)
        error['MSE'] = paddle.sum(paddle.pow(diff_mat, 2)) / n_valid_element
        error['RMSE'] = paddle.sqrt(error['MSE'])
        error['MAE'] = paddle.sum(diff_mat) / n_valid_element
        error['ABS_REL'] = paddle.sum(rel_mat) / n_valid_element
        y_over_z = paddle.divide(_gt_depth, _pred_depth)
        z_over_y = paddle.divide(_pred_depth, _gt_depth)
        max_ratio = paddle.maximum(y_over_z, z_over_y)
        error['DELTA1.02'] = paddle.sum(max_ratio < 1.02).numpy() / float(n_valid_element)
        error['DELTA1.05'] = paddle.sum(max_ratio < 1.05).numpy() / float(n_valid_element)
        error['DELTA1.10'] = paddle.sum(max_ratio < 1.10).numpy() / float(n_valid_element)
        error['DELTA1.25'] = paddle.sum(max_ratio < 1.25).numpy() / float(n_valid_element)
        error['DELTA1.25^2'] = paddle.sum(max_ratio < 1.25 ** 2).numpy() / float(n_valid_element)
        error['DELTA1.25^3'] = paddle.sum(max_ratio < 1.25 ** 3).numpy() / float(n_valid_element)

    error['MSE'] = error['MSE'].numpy()
    error['RMSE'] = error['RMSE'].numpy()
    error['ABS_REL'] = error['ABS_REL'].numpy()
    error['MAE'] = error['MAE'].numpy()
    return error


class Logger:
    def __init__(self, log_dir=None):
        from visualdl import LogWriter
        from datetime import datetime
        if log_dir is None:
            log_dir = f'./log/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = LogWriter(logdir=log_dir)

    def write_log(self, step, metrics, mode):
        for key in metrics.keys():
            self.writer.add_scalar(f'{mode}/{key}', metrics[key], step)

    def add_scalar(self, tag, value, step, walltime=None):
        self.writer.add_scalar(tag, value, step, walltime)

    def write_image(self, tag, image, step, **kwargs):
        self.writer.add_image(tag, image, step, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.writer.close()


def get_vis_depth_img(img):
    stretch = skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)).astype(np.uint8)
    stretch = cv2.merge([stretch, stretch, stretch])

    # define colors
    color1 = (0, 0, 255)  # red
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color5 = (255, 0, 0)  # blue
    color6 = (128, 64, 64)  # violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20, 1))
    grad = cv2.merge([grad, grad, grad])

    # apply lut to gradient for viewing
    grad_colored = cv2.LUT(grad, lut)

    return result, grad_colored


def get_out_img(pred_img, gt_img):
    if isinstance(pred_img, paddle.Tensor):
        pred_img = pred_img.numpy()
    if isinstance(gt_img, paddle.Tensor):
        gt_img = gt_img.numpy()

    pred_img, grad_colored = get_vis_depth_img(pred_img)
    gt_img, _ = get_vis_depth_img(gt_img)
    out_img = cv2.hconcat([gt_img, pred_img])
    return out_img
