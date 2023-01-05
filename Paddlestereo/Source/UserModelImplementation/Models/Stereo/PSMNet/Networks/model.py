# -*- coding: utf-8 -*-
import math
import paddle
from paddle import nn
from .submodule import convbn_3d, feature_extraction, disparityregression
import paddle.fluid.layers as layers
import paddle.fluid as fluid


class hourglass(nn.Layer):
    def __init__(self, inplanes: int) -> object:
        super().__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU())

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2,
                                             kernel_size=3, stride=2, pad=1),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2,
                                             kernel_size=3, stride=1, pad=1),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv3DTranspose(inplanes * 2, inplanes * 2,
                                                      kernel_size=3, padding=1, output_padding=1,
                                                      stride=2, bias_attr=False),
                                   nn.BatchNorm3D(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(nn.Conv3DTranspose(inplanes * 2, inplanes, kernel_size=3,
                                                      padding=1, output_padding=1, stride=2,
                                                      bias_attr=False),
                                   nn.BatchNorm3D(inplanes))  # +x

    def forward(self, x: paddle.tensor, presqu:
                paddle.tensor, postsqu: paddle.tensor) -> paddle.tensor:
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        m = paddle.nn.ReLU()
        if postsqu is not None:
            pre = m(pre + postsqu)
        else:
            pre = m(pre)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = m(self.conv5(out) + presqu)  # in:1/16 out:1/8
        else:
            post = m(self.conv5(out) + pre)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Layer):
    def __init__(self, maxdisp: int) -> object:
        super().__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv3D(32, 1, kernel_size=3,
                                                padding=1, stride=1, bias_attr=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv3D(32, 1, kernel_size=3,
                                                padding=1, stride=1, bias_attr=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv3D(32, 1, kernel_size=3,
                                                padding=1, stride=1, bias_attr=False))

        for m in self.parameters():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2D, nn.BatchNorm3D)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left: paddle.tensor, right: paddle.tensor) -> paddle.tensor:

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        cost = paddle.zeros([refimg_fea.shape[0],
                             refimg_fea.shape[1] * 2,
                             int(self.maxdisp / 4),
                             refimg_fea.shape[2],
                             refimg_fea.shape[3]])

        for i in range(int(self.maxdisp / 4)):
            if i > 0:
                cost[:, :refimg_fea.shape[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.shape[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.shape[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.shape[1]:, i, :, :] = targetimg_fea

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        upsample_out = paddle.nn.Upsample(size=[self.maxdisp, left.shape[2],
                                                left.shape[3]], data_format='NCDHW', mode='trilinear')
        if self.training:
            cost1 = upsample_out(cost1)
            cost2 = upsample_out(cost2)

            cost1 = layers.squeeze(input=cost1, axes=[1])
            pred1 = fluid.layers.softmax(cost1, axis=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = layers.squeeze(input=cost2, axes=[1])
            pred2 = fluid.layers.softmax(cost2, axis=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = upsample_out(cost3)
        cost3 = layers.squeeze(input=cost3, axes=[1])
        pred3 = fluid.layers.softmax(cost3, axis=1)
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3
