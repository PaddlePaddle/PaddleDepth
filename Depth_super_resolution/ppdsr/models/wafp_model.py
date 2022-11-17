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

import math

import paddle
import paddle.nn as nn

from .builder import MODELS
from .base_model import BaseModel

from .generators.builder import build_generator
from .criterions.builder import build_criterion
from ..utils.visual import tensor2img


@MODELS.register()
class WAFPModel(BaseModel):
    """WAFP-Net Model.

    Paper: WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for 
    Depth Map Super-resolution (IEEE Transactions on Multimedia 2021).
    https://ieeexplore.ieee.org/document/9563214/
    """
    def __init__(self, generator, mse_criterion=None, tv_criterion=None, scale=4, patch_size=61):
        """Initialize the WAFP class.

        Args:
            generator (dict): config of generator.
            char_criterion (dict): config of char criterion.
            edge_criterion (dict): config of edge criterion.
        """
        super(WAFPModel, self).__init__(generator)
        self.current_iter = 1
        self.scale = scale
        self.patch_size = patch_size
        self.nets['generator'] = build_generator(generator)

        if tv_criterion:
            self.tv_criterion = build_criterion(tv_criterion)
        if mse_criterion:
            self.mse_criterion = build_criterion(mse_criterion)

    def setup_input(self, input):
        self.target = input[1]
        self.lq = input[0]

    def train_iter(self, optims=None):
        optims['optim'].clear_gradients()

        out1, out2, out3, out = self.nets['generator'](self.lq)
        loss1 = self.mse_criterion(out1, self.target)
        loss2 = self.mse_criterion(out2, self.target)
        loss3 = self.mse_criterion(out3, self.target)
        loss5 = self.mse_criterion(out, self.target)
        tv_loss1 = self.tv_criterion(out)
        tv_loss2 = self.tv_criterion(out3)

        loss = 0.1 * loss1 + \
               0.2 * loss2 + \
               0.3 * loss3 + \
               0.4 * loss5 + \
               0.05 * tv_loss1 +\
               0.05 * tv_loss2


        loss.backward()
        optims['optim'].step()
        self.losses['loss'] = loss.numpy()

    def forward(self):
        pass

    def test_iter(self, metrics=None):
        self.nets['generator'].eval()
        with paddle.no_grad():
            self.output = self._pred_net_patch(self.lq, self.scale, self.patch_size)
            self.visual_items['output'] = self.output
        self.nets['generator'].train()

        out_img = []
        gt_img = []
        for out_tensor, gt_tensor in zip(self.output, self.target):
            out_img.append(tensor2img(out_tensor, (0., 1.)))
            gt_img.append(tensor2img(gt_tensor, (0., 1.)))

        if metrics is not None:
            for metric in metrics.values():
                metric.update(out_img, gt_img)

    def _pred_net(
        self, imgs: paddle.Tensor):
        return self.nets['generator'](imgs)[-1]

    def _pred_net_patch(self, imgs: paddle.Tensor, scale: int,
                           sub_s: int):
        h, w = imgs.shape[-2:]
        m = math.ceil(h / (sub_s))
        n = math.ceil(w / (sub_s))

        full_out = paddle.zeros_like(imgs, dtype='float32')  # [b,c,h,w]

        # process center
        for i in range(1, m):
            for j in range(1, n):
                begx = (i - 1) * sub_s - scale
                begy = (j - 1) * sub_s - scale
                endx = i * sub_s + scale
                endy = j * sub_s + scale
                if (begx < 0):
                    begx = 0
                if (begy < 0):
                    begy = 0
                if (endx > h):
                    endx = h
                if (endy > w):
                    endy = w

                im_input = imgs[:, :, begx:endx, begy:endy]
                out_patch = self._pred_net(im_input)
                out_patch = out_patch.detach()
                im_h_y = out_patch
                im_h_y = im_h_y * 255.0
                im_h_y = im_h_y.clip(0.0, 255.0)
                im_h_y = im_h_y / 255.0

                sh, sw = paddle.shape(im_h_y)[-2:]
                full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
                    im_h_y[:, :, scale:sh - scale, scale:sw - scale]

        # process edge
        for i in range(1, n):
            begx = h - sub_s - scale
            begy = (i - 1) * sub_s - scale
            endx = h
            endy = i * sub_s + scale
            if (begx < 0):
                begx = 0
            if (begy < 0):
                begy = 0
            if (endx > h):
                endx = h
            if (endy > w):
                endy = w
            im_input = imgs[:, :, begx:endx, begy:endy]
            out_patch = self._pred_net(im_input)
            out_patch = out_patch.detach()
            im_h_y = out_patch
            im_h_y = im_h_y * 255.0
            im_h_y = im_h_y.clip(0.0, 255.0)
            im_h_y = im_h_y / 255.0
            sh, sw = paddle.shape(im_h_y)[-2:]
            full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
                im_h_y[:, :, scale:sh - scale, scale:sw - scale]

        # process edge
        for i in range(1, m):
            begx = (i - 1) * sub_s - scale
            begy = w - sub_s - scale
            endx = i * sub_s + scale
            endy = w
            if (begx < 0):
                begx = 0
            if (begy < 0):
                begy = 0
            if (endx > h):
                endx = h
            if (endy > w):
                endy = w
            im_input = imgs[:, :, begx:endx, begy:endy]
            out_patch = self._pred_net(im_input)
            out_patch = out_patch.detach()
            im_h_y = out_patch
            im_h_y = im_h_y * 255.0
            im_h_y = im_h_y.clip(0.0, 255.0)
            im_h_y = im_h_y / 255.0
            sh, sw = paddle.shape(im_h_y)[-2:]
            full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
                im_h_y[:, :, scale:sh - scale, scale:sw - scale]
            im_input = im_input.detach()

        # process remain
        begx = h - sub_s - scale
        endx = h
        begy = w - sub_s - scale
        endy = w
        im_input = imgs[:, :, begx:endx, begy:endy]
        out_patch = self._pred_net(im_input)
        out_patch = out_patch.detach()
        im_h_y = out_patch
        im_h_y = im_h_y * 255.0
        im_h_y = im_h_y.clip(0.0, 255.0)
        im_h_y = im_h_y / 255.0
        sh, sw = paddle.shape(im_h_y)[-2:]
        full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
            im_h_y[:, :, scale:sh - scale, scale:sw - scale]

        return full_out