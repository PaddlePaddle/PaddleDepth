#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np

import paddle

from .builder import METRICS
from .psnr_ssim import reorder_image, PSNR



@METRICS.register()
class RMSE(PSNR):
    def update(self, preds, gts, is_seq=False):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        if is_seq:
            single_seq = []

        for pred, gt in zip(preds, gts):
            pred, gt = pred[:, :, 0:1], gt[:, :, 0:1]
            value = calculate_rmse(pred, gt, self.crop_border, self.input_order)
            if is_seq:
                single_seq.append(value)
            else:
                self.results.append(value)

        if is_seq:
            self.results.append(np.mean(single_seq))

    def name(self):
        return 'RMSE'


@METRICS.register()
class MAD(PSNR):
    def update(self, preds, gts, is_seq=False):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        if is_seq:
            single_seq = []

        for pred, gt in zip(preds, gts):
            pred, gt = pred[:, :, 0:1], gt[:, :, 0:1]
            value = calculate_mad(pred, gt, self.crop_border, self.input_order)
            if is_seq:
                single_seq.append(value)
            else:
                self.results.append(value)

        if is_seq:
            self.results.append(np.mean(single_seq))

    def name(self):
        return 'MAD'


@METRICS.register()
class PE(PSNR):
    def update(self, preds, gts, is_seq=False):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        if is_seq:
            single_seq = []

        for pred, gt in zip(preds, gts):
            pred, gt = pred[:, :, 0:1], gt[:, :, 0:1]
            value = calculate_pe(pred, gt, self.crop_border, self.input_order)
            if is_seq:
                single_seq.append(value)
            else:
                self.results.append(value)

        if is_seq:
            self.results.append(np.mean(single_seq))

    def name(self):
        return 'PE'



def calculate_rmse(img1,
                   img2,
                   crop_border,
                   input_order='HWC'):
    """Calculate RMSE (Root Mean Square Error).

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: rmse result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = img1.copy().astype('float32')
    img2 = img2.copy().astype('float32')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    diff = img1 - img2
    h, w = diff.shape[:2]
    mse = np.mean(diff**2)
    
    if mse == 0:
        return float('inf')
    rmse = np.sqrt(np.sum(np.power(diff, 2) / (h * w)))
    return rmse

def calculate_mad(img1,
                  img2,
                  crop_border,
                  input_order='HWC'):
    """Calculate MAD (Median Absolute Deviation).

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: mad result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = img1.copy().astype('float32')
    img2 = img2.copy().astype('float32')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    diff = img1 - img2
    h, w = diff.shape[:2]
    mad = np.sum(abs(diff)) / (h * w)
    
    if mad == 0:
        return float('inf')
    return mad

def calculate_pe(img1,
                  img2,
                  crop_border,
                  input_order='HWC'):
    """Calculate PE (Percentage of Error pixels).

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: PE result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = img1.copy().astype('float32')
    img2 = img2.copy().astype('float32')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    diff = img1 - img2
    h, w = diff.shape[:2]
    pe = np.sum(abs(diff)/ img2) / (h*w)
    
    if pe == 0:
        return float('inf')
    return pe

