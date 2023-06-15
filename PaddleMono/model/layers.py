# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import paddle.nn as nn
import paddle.nn.functional as F
import paddle


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose([0, 2, 1])
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = paddle.matmul(R, T)
    else:
        M = paddle.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = paddle.zeros([translation_vector.shape[0], 4, 4])

    t = paddle.reshape(translation_vector, [-1, 3, 1])

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """

    angle = paddle.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = paddle.cos(angle)
    sa = paddle.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = paddle.zeros((vec.shape[0], 4, 4))

    rot[:, 0, 0] = paddle.squeeze(x * xC + ca)
    rot[:, 0, 1] = paddle.squeeze(xyC - zs)
    rot[:, 0, 2] = paddle.squeeze(zxC + ys)
    rot[:, 1, 0] = paddle.squeeze(xyC + zs)
    rot[:, 1, 1] = paddle.squeeze(y * yC + ca)
    rot[:, 1, 2] = paddle.squeeze(yzC - xs)
    rot[:, 2, 0] = paddle.squeeze(zxC - ys)
    rot[:, 2, 1] = paddle.squeeze(yzC + xs)
    rot[:, 2, 2] = paddle.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

class Attention_net(nn.Layer):
    def __init__(self,in_c, out_c):
        super(Attention_net, self).__init__()
        self.K = nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.Q = nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.V = nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.local_weight = nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias_attr=False)

    def forward(self, x):
        k = self.K(x)
        v = self.V(x)
        q = self.Q(x)

        v_reshape = v.reshape([x.size(0), x.size(1), -1])
        v_reshape = v_reshape.transpose([0, 2, 1])
        q_reshape = q.reshape([x.size(0), x.size(1), -1])
        k_reshape = k.reshape([x.size(0), x.size(1), -1])
        k_reshape = k_reshape.transpose([0, 2, 1])

        qv = paddle.matmul(q_reshape, v_reshape)
        attention = F.softmax(qv, axis=-1)

        vector = paddle.matmul(k_reshape, attention)
        vector_reshape = paddle.transpose(vector, [0, 2, 1])

        O = vector_reshape.reshape([x.size(0), x.size(1), x.size(2), x.size(3)])
        out = paddle.add(x, O)
        out = self.local_weight(out)
        return out

class ConvBlock_modify(nn.Layer):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_modify, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock_attention(nn.Layer):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_attention, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2D(out_channels)
        self.attention = Attention_net(out_channels, out_channels)
        self.nonlin = nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.attention(out)
        out = self.nonlin(out)
        return out

class ConvBlock(nn.Layer):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Layer):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.Pad2D(1, mode='reflect')
        else:
            self.pad = nn.Pad2D(1)
        self.conv = nn.Conv2D(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Layer):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        constant_attr = paddle.ParamAttr(trainable=False)
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = self.create_parameter(paddle.to_tensor(id_coords).shape,
                                               attr=constant_attr,
                                               default_initializer=nn.initializer.Assign(id_coords))

        self.ones = self.create_parameter([self.batch_size, 1, self.height * self.width],
                                 attr=constant_attr,
                                 default_initializer=nn.initializer.Constant(1))

        pix_coords = paddle.unsqueeze(paddle.stack(
            [self.id_coords[0].reshape([-1]), self.id_coords[1].reshape([-1])], 0), 0)
        pix_coords = pix_coords.tile([batch_size, 1, 1])
        self.pix_coords = self.create_parameter(paddle.concat([pix_coords, self.ones], 1).shape,
                                                attr=constant_attr,
                                                default_initializer=paddle.nn.initializer.Assign(paddle.concat([pix_coords, self.ones], 1)))

    def forward(self, depth, inv_K):
        cam_points = paddle.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.reshape([self.batch_size, 1, -1]) * cam_points
        cam_points = paddle.concat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Layer):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = paddle.matmul(K, T)[:, :3, :]

        cam_points = paddle.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.reshape([self.batch_size, 2, self.height, self.width])
        pix_coords = pix_coords.transpose([0, 2, 3, 1])
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness losses for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = paddle.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = paddle.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = paddle.mean(paddle.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = paddle.mean(paddle.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= paddle.exp(-grad_img_x)
    grad_disp_y *= paddle.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class silog_loss(nn.Layer):
    """The classification metric based on probability
    """
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = paddle.log(depth_est[mask]) - paddle.log(depth_gt[mask])
        return paddle.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class SSIM(nn.Layer):
    """Layer to compute the SSIM losses between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2D(3, 1)
        self.mu_y_pool   = nn.AvgPool2D(3, 1)
        self.sig_x_pool  = nn.AvgPool2D(3, 1)
        self.sig_y_pool  = nn.AvgPool2D(3, 1)
        self.sig_xy_pool = nn.AvgPool2D(3, 1)

        self.refl = nn.Pad2D(1, mode="reflect")

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return paddle.clip((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt = gt.numpy()
    pred = pred.numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
