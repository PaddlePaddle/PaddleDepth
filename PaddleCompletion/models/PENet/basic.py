import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def get_pads(kernel_size: int = 5):
    """
    Returns a list of zero padding layers, where each layer has a different
    amount of padding on each side. The amount of padding is determined by
    the kernel size and the position of the layer in the list.

    Args:
        kernel_size (int): The size of the guide kernel.

    Returns:
        List[paddle.nn.ZeroPad2D]: A list of zero padding layers.
    """
    pad = [i for i in range(kernel_size * kernel_size)]
    for i in range(kernel_size):
        for j in range(kernel_size):
            top = i
            bottom = kernel_size - 1 - i
            left = j
            right = kernel_size - 1 - j
            pad[i * kernel_size + j] = nn.ZeroPad2D([left, right, top, bottom])
    return pad


def weights_init(m: nn.Layer):
    """
    Initializes the weights of the given layer using Gaussian random weights.
    For convolutional and transposed convolutional layers, the weights are
    initialized using the formula sqrt(2/n), where n is the number of weights
    in the kernel. For batch normalization layers, the weights are initialized
    to 1 and the biases to 0.

    Args:
        m (paddle.nn.Layer): The layer to initialize the weights of.
    """
    if isinstance(m, nn.Conv2D):
        n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
        m.weight.set_value(paddle.randn(m.weight.shape) * math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.set_value(paddle.zeros(m.bias.shape))
    elif isinstance(m, nn.Conv2DTranspose):
        n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
        m.weight.set_value(paddle.randn(m.weight.shape) * math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.set_value(paddle.zeros(m.bias.shape))
    elif isinstance(m, nn.BatchNorm2D):
        m.weight.set_value(paddle.ones(m.weight.shape))
        m.bias.set_value(paddle.zeros(m.bias.shape))


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False,
        ),
        nn.BatchNorm2D(out_channels),
        nn.ReLU(),
    )


def deconvbnrelu(
    in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
):
    return nn.Sequential(
        nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias_attr=False,
        ),
        nn.BatchNorm2D(out_channels),
        nn.ReLU(),
    )


def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False,
        ),
        nn.BatchNorm2D(out_channels),
    )


def deconvbn(
    in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0
):
    return nn.Sequential(
        nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias_attr=False,
        ),
        nn.BatchNorm2D(out_channels),
    )


class BasicBlock(nn.Layer):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(
    in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1
):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias_attr=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias_attr=bias,
    )


class SparseDownSampleClose(nn.Layer):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2D(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = -(1 - mask) * self.large_number - d

        d = -self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number

        return d_result, mask_result


class CSPNGenerate(nn.Layer):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(
            in_channels,
            self.kernel_size * self.kernel_size - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, feature):
        guide = self.generate(feature)

        # normalization
        guide_sum = paddle.sum(paddle.abs(guide), axis=1).unsqueeze(1)
        guide = guide / guide_sum
        guide_mid = (1 - paddle.sum(guide, axis=1)).unsqueeze(1)

        # padding
        weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if self.kernel_size == 3:
                zero_pad = get_pads(3)[t]
            elif self.kernel_size == 5:
                zero_pad = get_pads(5)[t]
            elif self.kernel_size == 7:
                zero_pad = get_pads(7)[t]
            if t < int((self.kernel_size * self.kernel_size - 1) / 2):
                weight_pad[t] = zero_pad(guide[:, t : t + 1, :, :])
            elif t > int((self.kernel_size * self.kernel_size - 1) / 2):
                weight_pad[t] = zero_pad(guide[:, t - 1 : t, :, :])
            else:
                weight_pad[t] = zero_pad(guide_mid)

        guide_weight = paddle.concat(
            [weight_pad[t] for t in range(self.kernel_size * self.kernel_size)], axis=1
        )
        return guide_weight


class CSPN(nn.Layer):
    def __init__(self, kernel_size):
        super(CSPN, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, guide_weight, hn, h0):
        # CSPN
        half = int(0.5 * (self.kernel_size * self.kernel_size - 1))
        result_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if self.kernel_size == 3:
                zero_pad = get_pads(3)[t]
            elif self.kernel_size == 5:
                zero_pad = get_pads(5)[t]
            elif self.kernel_size == 7:
                zero_pad = get_pads(7)[t]
            if t == half:
                result_pad[t] = zero_pad(h0)
            else:
                result_pad[t] = zero_pad(hn)
        guide_result = paddle.concat(
            [result_pad[t] for t in range(self.kernel_size * self.kernel_size)], axis=1
        )

        guide_result = paddle.sum((guide_weight * guide_result), axis=1)
        guide_result = guide_result[
            :,
            int((self.kernel_size - 1) / 2) : -int((self.kernel_size - 1) / 2),
            int((self.kernel_size - 1) / 2) : -int((self.kernel_size - 1) / 2),
        ]

        return paddle.unsqueeze(guide_result, axis=1)


class CSPNGenerateAccelerate(nn.Layer):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(
            in_channels, self.kernel_size * self.kernel_size - 1, 3, 1, 1
        )

    def forward(self, feature):
        guide = self.generate(feature)

        guide_sum = paddle.sum(paddle.abs(guide), axis=1).unsqueeze(1)
        guide = paddle.divide(guide, guide_sum)
        guide_mid = (1 - paddle.sum(guide, axis=1)).unsqueeze(1)

        half1, half2 = paddle.chunk(guide, chunks=2, axis=1)
        output = paddle.concat([half1, guide_mid, half2], axis=1)
        return output


def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.shape[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size - 1) / 2))
    return kernel


class CSPNAccelerate(nn.Layer):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0):
        bs = input.shape[0]
        h, w = input.shape[2], input.shape[3]
        input_im2col = F.unfold(
            input, self.kernel_size, self.stride, self.padding, self.dilation
        )
        kernel = kernel.reshape([bs, self.kernel_size * self.kernel_size, h * w])

        input0 = input0.reshape([bs, 1, h * w])
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        input_im2col[:, mid_index : mid_index + 1, :] = input0

        output = paddle.einsum("ijk,ijk->ik", input_im2col, kernel)
        return output.reshape([bs, 1, h, w])


class GeometryFeature(nn.Layer):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return paddle.concat([x, y, z], axis=1)


class BasicBlockGeo(nn.Layer):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        geoplanes=3,
    ):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
            # norm_layer = encoding.nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes + geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = paddle.concat((x, g1), axis=1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = paddle.concat((g2, out), axis=1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
