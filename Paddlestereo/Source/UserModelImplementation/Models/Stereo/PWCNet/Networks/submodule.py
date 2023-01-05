from __future__ import print_function
import paddle
from paddle import nn


class Mish(nn.Layer):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        # save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x * (paddle.tanh(nn.functional.softplus(x)))


def convbn(in_planes: int, out_planes: int,
           kernel_size: int, stride: int, pad: int, dilation: int) -> object:
    return nn.Sequential(nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=dilation if dilation > 1 else pad,
                                   dilation=dilation, bias_attr=False),
                         nn.BatchNorm2D(out_planes))


def convbn_3d(in_planes: int, out_planes: int, kernel_size: int,
              stride: int, pad: int) -> object:
    return nn.Sequential(nn.Conv3D(in_planes, out_planes, kernel_size=kernel_size,
                                   padding=pad, stride=stride, bias_attr=False),
                         nn.BatchNorm3D(out_planes))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = paddle.arange(0, maxdisp, dtype=x.dtype)
    disp_values = paddle.reshape(disp_values, shape=[1, maxdisp, 1, 1])
    return paddle.sum(x * disp_values, 1, keepdim=False)


def disp_regression_nearby(similarity, disp_step, half_support_window=2):
    """Returns predicted disparity with subpixel_map(disp_similarity).

    Predicted disparity is computed as:

    d_predicted = sum_d( d * P_predicted(d)),
    where | d - d_similarity_maximum | < half_size

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        half_support_window: defines size of disparity window in pixels
                             around disparity with maximum similarity,
                             which is used to convert similarities
                             to probabilities and compute mean.
    """

    assert 4 == similarity.dim(), \
        'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())

    # In every location (x, y) find disparity with maximum similarity score.
    similar_maximum, idx_maximum = paddle.max(similarity, axis=1, keepdim=True)
    idx_limit = similarity.size(1) - 1

    # Collect similarity scores for the disparities around the disparity
    # with the maximum similarity score.
    support_idx_disp = []
    for idx_shift in range(-half_support_window, half_support_window + 1):
        idx_disp = idx_maximum + idx_shift
        idx_disp[idx_disp < 0] = 0
        idx_disp[idx_disp >= idx_limit] = idx_limit
        support_idx_disp.append(idx_disp)

    support_idx_disp = paddle.concat(support_idx_disp, axis=1)
    support_similar = paddle.gather(similarity, support_idx_disp.long(), 1)
    support_disp = support_idx_disp.float() * disp_step

    # Convert collected similarity scores to the disparity distribution
    # using softmax and compute disparity as a mean of this distribution.
    prob = nn.functional.softmax(support_similar, axis=1)
    disp = paddle.sum(prob * support_disp.float(), axis=1)

    return disp


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = paddle.zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    # volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = fea1 * fea2
    cost = paddle.reshape(cost, shape=[B, num_groups, channels_per_group, H, W])
    cost = paddle.mean(cost, axis=2)
    # assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = paddle.zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    # volume = volume.contiguous()
    return volume


def build_corrleation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = paddle.zeros([B, num_groups, 2 * maxdisp + 1, H, W])

    for i in range(-maxdisp, maxdisp + 1):
        if i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                                     num_groups)
        elif i < 0:
            volume[:, :, i + maxdisp, :, :-i] = groupwise_correlation(refimg_fea[:, :, :, :-i],
                                                                      targetimg_fea[:, :, :, i:],
                                                                      num_groups)
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    # volume = volume.contiguous()
    return volume


def warp(x, disp):
    """
    warp an image/tensor (imright) back to imleft, according to the disp

    x: [B, C, H, W] (imright)
    disp: [B, 1, H, W] disp

    """
    B, C, H, W = x.shape
    # device = x.get_device()
    # mesh grid
    xx = paddle.repeat_interleave(paddle.reshape(paddle.arange(0, W, dtype='float32'), shape=[1, -1]), H, 0)
    yy = paddle.repeat_interleave(paddle.reshape(paddle.arange(0, H, dtype='float32'), shape=[-1, 1]), W, 1)

    xx = paddle.repeat_interleave(paddle.reshape(xx, shape=[1, 1, H, W]), B, 0)
    yy = paddle.repeat_interleave(paddle.reshape(yy, shape=[1, 1, H, W]), B, 0)
    # grid = torch.cat((xx, yy), 1).float()

#     if x.is_cuda:
#         xx = xx.float().cuda()
#         yy = yy.float().cuda()
    xx_warp = xx - disp
#     xx_warp = xx - disp
    vgrid = paddle.concat((xx_warp, yy), 1)
    # vgrid = Variable(grid) + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = paddle.transpose(vgrid, (0, 2, 3, 1))
    output = nn.functional.grid_sample(x, vgrid)
    mask = paddle.ones(x.shape)
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask


def FMish(x):
    '''

    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''
    return x * (paddle.tanh(nn.functional.softplus(x)))


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   Mish())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
