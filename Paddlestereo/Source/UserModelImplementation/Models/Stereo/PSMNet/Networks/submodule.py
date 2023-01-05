# -*- coding: utf-8 -*-
import paddle
from paddle import nn
import numpy as np


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


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int,
                 downsample: bool, pad: int, dilation: int) -> object:
        super().__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Layer):
    def __init__(self, maxdisp: int) -> object:
        super().__init__()
        self.disp = paddle.to_tensor(np.reshape(np.array(range(maxdisp)),
                                                [1, maxdisp, 1, 1]), stop_gradient=True)

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        disp = paddle.tile(self.disp, [x.shape[0], 1, x.shape[2], x.shape[3]])
        return paddle.sum(x * disp, 1)


class feature_extraction(nn.Layer):
    def __init__(self) -> object:
        super().__init__()
        self.inplanes = 32
        self.firstconv = self._first_layer()

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2D((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch2 = nn.Sequential(nn.AvgPool2D((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch3 = nn.Sequential(nn.AvgPool2D((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.branch4 = nn.Sequential(nn.AvgPool2D((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU())

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv2D(128, 32, kernel_size=1,
                                                padding=0, stride=1, bias_attr=False))

    def _first_layer(self) -> object:
        return nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                             nn.ReLU(),
                             convbn(32, 32, 3, 1, 1, 1),
                             nn.ReLU(),
                             convbn(32, 32, 3, 1, 1, 1),
                             nn.ReLU())

    def _make_layer(self, block: object, planes: int, blocks: int,
                    stride: int, pad: int, dilation: int) -> object:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),)

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x: paddle.tensor) -> paddle.tensor:
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        upsample_out = paddle.nn.Upsample(size=[output_skip.shape[2],
                                                output_skip.shape[3]], mode='bilinear')
        output_branch1 = upsample_out(output_branch1)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = upsample_out(output_branch2)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = upsample_out(output_branch3)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = upsample_out(output_branch4)

        output_feature = paddle.concat((output_raw, output_skip, output_branch4,
                                        output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
