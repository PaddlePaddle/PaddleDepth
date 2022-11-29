# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import numpy as np
from collections import OrderedDict
from .layers import *
import paddle
from paddle.nn import functional as F
import paddle.nn as nn


class Attention_net(nn.Layer):
    def __init__(self, in_c, out_c):
        super(Attention_net, self).__init__()
        self.K = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1,bias_attr=False)
        self.Q = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1,bias_attr=False)
        self.V = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1,bias_attr=False)
        self.local_weight = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0,bias_attr=False)

    def forward(self, x):
        k = self.K(x)
        v = self.V(x)
        q = self.Q(x)

        v_reshape = v.reshape([x.shape[0], x.shape[1], -1])
        v_reshape = v_reshape.transpose([0, 2, 1])
        q_reshape = q.reshape([x.shape[0], x.shape[1], -1])
        k_reshape = k.reshape([x.shape[0], x.shape[1], -1])
        k_reshape = k_reshape.transpose([0, 2, 1])

        qv = paddle.matmul(q_reshape, v_reshape)
        attention = F.softmax(qv, axis=-1)

        vector = paddle.matmul(k_reshape, attention)
        vector_reshape = paddle.transpose(vector, [0, 2, 1])

        O = vector_reshape.reshape([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        out = paddle.add(x, O)
        out = self.local_weight(out)
        return out


def diff_x(input, r):
    assert len(input.shape) == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

    output = paddle.concat([left, middle, right], axis=2)

    return output

def diff_y(input, r):
    assert len(input.shape) == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = paddle.concat([left, middle, right], axis=3)

    return output


class BoxFilter(nn.Layer):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert len(x.shape) == 4

        return diff_y(diff_x(x.cumsum(axis=2), self.r).cumsum(axis=3), self.r)

class GridAttentionBlock(nn.Layer):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2D(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1)

        self.phi = nn.Conv2D(in_channels=self.gating_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2D(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, g):
        input_size = x.shape
        batch_size = input_size[0]
        assert batch_size == g.shape[0]

        theta_x = self.theta(x)
        theta_x_size = theta_x.shape

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g)

        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f


class FastGuidedFilter_attention(nn.Layer):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.shape
        n_lry, c_lry, h_lry, w_lry = lr_y.shape
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.shape

        lr_x = lr_x.astype('float64')
        lr_y = lr_y.astype('float64')
        hr_x = hr_x.astype('float64')
        l_a = l_a.astype('float64')

        # if i == 2:


        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        mid_N = paddle.ones([1, 1, h_lrx, w_lrx], dtype='float64')
        N = self.boxfilter(paddle.to_tensor(mid_N))

        l_a = paddle.abs(l_a) + self.epss


        t_all = paddle.sum(l_a)

        l_t = l_a / t_all

        mean_a = self.boxfilter(l_a) / N
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        mean_ay = self.boxfilter(l_a * lr_y) / N
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        mean_ax = self.boxfilter(l_a * lr_x) / N

        temp = paddle.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (mean_A * hr_x + mean_b).astype('float32')


class DepthDecoderAttention_edge(nn.Layer):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoderAttention_edge, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # decoder
        self.convs = OrderedDict()

        self.convs[("fg")] = FastGuidedFilter_attention(r=2, eps=1e-2)

        # attention blocks
        self.convs[("edge", 4)] = GridAttentionBlock(in_channels=256)
        self.convs[("edge", 3)] = GridAttentionBlock(in_channels=128)
        self.convs[("edge", 2)] = GridAttentionBlock(in_channels=64)
        self.convs[("edge", 1)] = GridAttentionBlock(in_channels=32)
        self.convs[("edge", 0)] = GridAttentionBlock(in_channels=16)

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock_attention(num_ch_in, num_ch_out)
            self.convs[("upconv", i, 2)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.LayerList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x_tmp00 = x
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = paddle.concat(x, 1)
            x = self.convs[("upconv", i, 2)](x)

            x_tmp01 = F.interpolate(x, scale_factor=1/2, mode='bilinear')
            mid_x = self.convs[("edge", i)](x_tmp01, x_tmp00)
            x = self.convs[("fg")](x_tmp01, x_tmp00, x, mid_x)

            x_tmp = x

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x_tmp))
        return self.outputs
