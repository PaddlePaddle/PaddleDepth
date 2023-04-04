import paddle
from paddle import nn
from ext_ops import GuideConv
import math
from . import params_init


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False
    )


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias_attr=False,
        dilation=dilation,
    )


class Basic2D(nn.Layer):
    def __init__(
        self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1
    ):
        super(Basic2D, self).__init__()
        if norm_layer:
            conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias_attr=False,
            )
        else:
            conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias_attr=True,
            )
        self.conv = nn.Sequential(
            conv,
        )
        if norm_layer:
            self.conv.add_sublayer("bn", norm_layer(out_channels))
        self.conv.add_sublayer("relu", nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2DTrans(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(Basic2DTrans, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self.conv = nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias_attr=False,
        )
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Basic2DLocal(nn.Layer):
    def __init__(self, out_channels, norm_layer=None):
        super(Basic2DLocal, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.conv = GuideConv
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input, weight):
        out = self.conv(input, weight)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Guide(nn.Layer):
    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super(Guide, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.local = Basic2DLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2D((1, 1))
        self.conv11 = Basic2D(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2D(
            input_planes,
            input_planes * 9,
            kernel_size=weight_ks,
            padding=weight_ks // 2,
        )
        self.conv21 = Basic2D(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2D(
            input_planes, input_planes * input_planes, kernel_size=1, padding=0
        )
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(),
        )
        self.conv3 = Basic2D(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        (B, Ci, H, W) = input.shape
        weight = paddle.concat([input, weight], 1)
        weight11 = self.conv11(weight)
        weight12 = self.conv12(weight11)
        weight21 = self.conv21(weight)
        weight21 = self.pool(weight21)
        weight22 = self.conv22(weight21).reshape((B, -1, Ci))
        out = self.local(input, weight12).reshape((B, Ci, -1))
        out = paddle.bmm(weight22, out).reshape((B, Ci, H, W))
        out = self.br(out)
        out = self.conv3(out)
        return out


class BasicBlock(nn.Layer):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

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
        if self.act:
            out = self.relu(out)
        return out


class GuideNet(nn.Layer):
    """
    Not activate at the ref
    Init change to trunctated norm
    """

    def __init__(
        self,
        block=BasicBlock,
        bc=16,
        img_layers=[2, 2, 2, 2, 2],
        depth_layers=[2, 2, 2, 2, 2],
        norm_layer=nn.BatchNorm2D,
        guide=Guide,
        weight_ks=3,
    ):
        super(GuideNet, self).__init__()
        self._norm_layer = norm_layer

        self.conv_img = Basic2D(
            3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2
        )
        in_channels = bc * 2
        self.inplanes = in_channels
        self.layer1_img = self._make_layer(
            block, in_channels * 2, img_layers[0], stride=2
        )

        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img = self._make_layer(
            block, in_channels * 4, img_layers[1], stride=2
        )

        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img = self._make_layer(
            block, in_channels * 8, img_layers[2], stride=2
        )

        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img = self._make_layer(
            block, in_channels * 8, img_layers[3], stride=2
        )

        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img = self._make_layer(
            block, in_channels * 8, img_layers[4], stride=2
        )

        self.layer2d_img = Basic2DTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d_img = Basic2DTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d_img = Basic2DTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d_img = Basic2DTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv_lidar = Basic2D(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(
            block, in_channels * 2, depth_layers[0], stride=2
        )
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_lidar = self._make_layer(
            block, in_channels * 4, depth_layers[1], stride=2
        )
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_lidar = self._make_layer(
            block, in_channels * 8, depth_layers[2], stride=2
        )
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_lidar = self._make_layer(
            block, in_channels * 8, depth_layers[3], stride=2
        )
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_lidar = self._make_layer(
            block, in_channels * 8, depth_layers[4], stride=2
        )

        self.layer1d = Basic2DTrans(in_channels * 2, in_channels, norm_layer)
        self.layer2d = Basic2DTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d = Basic2DTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d = Basic2DTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d = Basic2DTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv = nn.Conv2D(bc * 2, 1, kernel_size=3, stride=1, padding=1)
        self.ref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)

        self._initialize_weights()

    def forward(self, img, lidar):
        c0_img = self.conv_img(img)
        c1_img = self.layer1_img(c0_img)
        c2_img = self.layer2_img(c1_img)
        c3_img = self.layer3_img(c2_img)
        c4_img = self.layer4_img(c3_img)
        c5_img = self.layer5_img(c4_img)
        dc5_img = self.layer5d_img(c5_img)
        c4_mix = dc5_img + c4_img
        dc4_img = self.layer4d_img(c4_mix)
        c3_mix = dc4_img + c3_img
        dc3_img = self.layer3d_img(c3_mix)
        c2_mix = dc3_img + c2_img
        dc2_img = self.layer2d_img(c2_mix)
        c1_mix = dc2_img + c1_img

        c0_lidar = self.conv_lidar(lidar)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar_dyn = self.guide1(c1_lidar, c1_mix)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)
        c2_lidar_dyn = self.guide2(c2_lidar, c2_mix)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_mix)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_mix)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)
        c5 = c5_img + c5_lidar
        dc5 = self.layer5d(c5)
        c4 = dc5 + c4_lidar_dyn
        dc4 = self.layer4d(c4)
        c3 = dc4 + c3_lidar_dyn
        dc3 = self.layer3d(c3)
        c2 = dc3 + c2_lidar_dyn
        dc2 = self.layer2d(c2)
        c1 = dc2 + c1_lidar_dyn
        dc1 = self.layer1d(c1)
        c0 = dc1 + c0_lidar
        output = self.ref(c0)
        output = self.conv(output)
        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for _, m in self.named_children():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._in_channels
                params_init.truncated_normal(
                    m.weight, mean=0, std=math.sqrt(1.3 * 2.0 / n)
                )
                params_init.constant_init(m.bias)


def GN():
    return GuideNet(norm_layer=nn.SyncBatchNorm, guide=Guide)


def GNS():
    return GuideNet(norm_layer=nn.SyncBatchNorm, guide=Guide, weight_ks=1)
