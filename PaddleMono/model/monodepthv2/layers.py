import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """
    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """
    Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation

    if invert:
        R = R.transpose((0, 2, 1))
        t = - t

    T = get_translation_matrix(t)

    if invert:
        M = paddle.matmul(R, T)
    else:
        M = paddle.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    T = paddle.zeros((translation_vector.shape[0], 4, 3))

    t = translation_vector.reshape((-1, 3, 1))
    t = paddle.concat([t, paddle.ones((t.shape[0], 1, 1))], axis=1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T = paddle.concat([T, t], axis=-1)

    return T


def rot_from_axisangle(vec):
    """
    Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    batch_size = vec.shape[0]
    angle = paddle.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)
    
    ca = paddle.cos(angle).squeeze()
    sa = paddle.sin(angle).squeeze()
    C = 1 - ca

    x = axis[..., 0].squeeze()
    y = axis[..., 1].squeeze(1)
    z = axis[..., 2].squeeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    ones = paddle.ones((batch_size,))
    zeros = paddle.zeros((batch_size,))
    rot = paddle.stack([
        paddle.stack([x * xC + ca, xyC - zs, zxC + ys, zeros], axis=1),
        paddle.stack([xyC + zs, y * yC + ca, yzC - xs, zeros], axis=1),
        paddle.stack([zxC - ys, yzC + xs, z * zC + ca, zeros], axis=1),
        paddle.stack([zeros, zeros, zeros, ones], axis=1)
    ], axis=1)

    return rot


class ConvBlock(nn.Layer):
    """
    Layer to perform a convolution followed by ELU
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
    """
    Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2D(int(in_channels), int(out_channels), 3, padding=1, padding_mode='reflect' if use_refl else 'zeros', weight_attr=nn.initializer.KaimingUniform())

    def forward(self, x):
        out = self.conv(x)
        return out


class BackprojectDepth(nn.Layer):
    """
    Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.register_buffer('id_coords', paddle.to_tensor(id_coords))

        self.register_buffer('ones', paddle.ones((self.batch_size, 1, self.height * self.width)))

        pix_coords = paddle.unsqueeze(paddle.stack([self.id_coords[0].reshape((-1,)), self.id_coords[1].reshape((-1,))], 0), 0)
        pix_coords = pix_coords.expand((batch_size, -1, -1))
        self.register_buffer('pix_coords', paddle.concat([pix_coords, self.ones], 1))

    def forward(self, depth, inv_K):
        cam_points = paddle.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.reshape((self.batch_size, 1, -1)) * cam_points
        cam_points = paddle.concat([cam_points, self.ones], axis=1)

        return cam_points


class Project3D(nn.Layer):
    """
    Layer which projects 3D points into a camera with intrinsics K and at position T
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
        pix_coords = pix_coords.reshape((self.batch_size, 2, self.height, self.width))
        pix_coords = pix_coords.transpose((0, 2, 3, 1))
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = paddle.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = paddle.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = paddle.mean(paddle.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = paddle.mean(paddle.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= paddle.exp(-grad_img_x)
    grad_disp_y *= paddle.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Layer):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2D(3, 1)
        self.mu_y_pool   = nn.AvgPool2D(3, 1)
        self.sig_x_pool  = nn.AvgPool2D(3, 1)
        self.sig_y_pool  = nn.AvgPool2D(3, 1)
        self.sig_xy_pool = nn.AvgPool2D(3, 1)

        self.refl = nn.Pad2D(1, mode='reflect')

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
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = paddle.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).astype(paddle.float32).mean()
    a2 = (thresh < 1.25 ** 2).astype(paddle.float32).mean()
    a3 = (thresh < 1.25 ** 3).astype(paddle.float32).mean()

    rmse = (gt - pred) ** 2
    rmse = paddle.sqrt(rmse.mean())

    rmse_log = (paddle.log(gt) - paddle.log(pred)) ** 2
    rmse_log = paddle.sqrt(rmse_log.mean())

    abs_rel = paddle.mean(paddle.abs(gt - pred) / gt)

    sq_rel = paddle.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
