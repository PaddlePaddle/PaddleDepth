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

import paddle
import paddle.nn as nn

from .generators.builder import build_generator
from .criterions.builder import build_criterion
from .base_model import BaseModel
from .builder import MODELS
from ..utils.visual import tensor2img
from ..modules.init import reset_parameters


@MODELS.register()
class PMBAModel(BaseModel):
    """Base SR model for single image super-resolution.
    """
    def __init__(self, generator, pixel_criterion=None, use_init_weight=False, window_size=16, scale=4):
        """
        Args:
            generator (dict): config of generator.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(PMBAModel, self).__init__()

        self.nets['generator'] = build_generator(generator)

        if pixel_criterion:
            self.pixel_criterion = build_criterion(pixel_criterion)
        if use_init_weight:
            init_sr_weight(self.nets['generator'])
        
        self.window_size = window_size
        self.scale = scale

    def setup_input(self, input):
        self.lq = paddle.to_tensor(input['lq'])[:, 0:1, :, :]
        self.visual_items['lq'] = self.lq
        if 'gt' in input:
            self.gt = paddle.to_tensor(input['gt'])[:, 0:1, :, :]
            self.visual_items['gt'] = self.gt
        self.image_paths = input['lq_path']

    def forward(self):
        pass

    def train_iter(self, optims=None):
        optims['optim'].clear_grad()

        self.output = self.nets['generator'](self.lq)
        self.visual_items['output'] = self.output
        # pixel loss
        loss_pixel = self.pixel_criterion(self.output, self.gt)
        self.losses['loss_pixel'] = loss_pixel

        loss_pixel.backward()
        optims['optim'].step()

    def test_iter(self, metrics=None):
        self.nets['generator'].eval()
        img_lq = self.lq
        _, _, h_old, w_old = img_lq.shape
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
        img_lq = paddle.concat([img_lq, paddle.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = paddle.concat([img_lq, paddle.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        with paddle.no_grad():
            output = self.nets['generator'](img_lq)
            self.output = output[..., :h_old * self.scale, :w_old * self.scale]
            self.visual_items['output'] = self.output
        self.nets['generator'].train()

        out_img = []
        gt_img = []
        for out_tensor, gt_tensor in zip(self.output, self.gt):
            out_img.append(tensor2img(out_tensor, (0., 1.)))
            gt_img.append(tensor2img(gt_tensor, (0., 1.)))

        if metrics is not None:
            for metric in metrics.values():
                metric.update(out_img, gt_img)


def init_sr_weight(net):
    def reset_func(m):
        if hasattr(m, 'weight') and (not isinstance(
                m, (nn.BatchNorm, nn.BatchNorm2D))):
            reset_parameters(m)

    net.apply(reset_func)
