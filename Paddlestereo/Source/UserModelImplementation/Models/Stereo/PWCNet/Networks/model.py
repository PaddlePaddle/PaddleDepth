from __future__ import print_function
import paddle
from paddle import nn
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from .submodule import FMish, build_gwc_volume, build_concat_volume, build_corrleation_volume
from .submodule import convbn, BasicBlock, Mish, convbn_3d, disparity_regression, warp
import math


class feature_extraction(nn.Layer):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer5 = self._make_layer(BasicBlock, 192, 3, 2, 1, 1)
        # self.layer6 = self._make_layer(BasicBlock, 192, 3, 1, 1, 1)
        self.layer7 = self._make_layer(BasicBlock, 256, 3, 2, 1, 1)
        # self.layer8 = self._make_layer(BasicBlock, 256, 3, 1, 1, 1)
        self.layer9 = self._make_layer(BasicBlock, 512, 3, 2, 1, 1)
        # self.layer10 = self._make_layer(BasicBlock, 512, 3, 1, 1, 1)

        self.gw2 = nn.Sequential(convbn(192, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2D(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias_attr=False))

        self.gw3 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2D(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias_attr=False))

        self.gw4 = nn.Sequential(convbn(512, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2D(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias_attr=False))

        self.layer11 = nn.Sequential(convbn(320, 320, 3, 1, 1, 1),
                                     Mish(),
                                     nn.Conv2D(320, 320, kernel_size=1, padding=0, stride=1,
                                               bias_attr=False))
        self.layer_refine = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          Mish(),
                                          convbn(128, 32, 1, 1, 0, 1),
                                          Mish())
        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2D(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias_attr=False))

            self.concat2 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2D(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias_attr=False))
            self.concat3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2D(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias_attr=False))

            self.concat4 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2D(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias_attr=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)  # 1/4
        l5 = self.layer5(l4)  # 1/8
        # l6 = self.layer6(l5)
        l6 = self.layer7(l5)  # 1/16
        # l8 = self.layer8(l7)
        l7 = self.layer9(l6)  # 1/32
        # l10 = self.layer10(l9)

        featurecombine = paddle.concat((l2, l3, l4), axis=1)
        # combine1 = torch.cat((l5, l6), dim=1)
        # combine2 = torch.cat((l7, l8), dim=1)
        # combine3 = torch.cat((l9, l10), dim=1)
        gw1 = self.layer11(featurecombine)
        gw2 = self.gw2(l5)
        gw3 = self.gw3(l6)
        gw4 = self.gw4(l7)
        feature_refine = self.layer_refine(featurecombine)

        if not self.concat_feature:
            return {"gw1": gw1, "gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            concat_feature1 = self.lastconv(featurecombine)
            concat_feature2 = self.concat2(l5)
            concat_feature3 = self.concat3(l6)
            concat_feature4 = self.concat4(l7)
            return {"gw1": gw1, "gw2": gw2, "gw3": gw3, "gw4": gw4, "concat_feature1": concat_feature1,
                    "finetune_feature": feature_refine,
                    "concat_feature2": concat_feature2,
                    "concat_feature3": concat_feature3,
                    "concat_feature4": concat_feature4}


class hourglassup(nn.Layer):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        # self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
        #                            Mish())
        self.conv1 = nn.Conv3D(in_channels, in_channels * 2, kernel_size=3, stride=2,
                               padding=1, bias_attr=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        # self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
        #                            Mish())
        self.conv3 = nn.Conv3D(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias_attr=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        # self.conv5 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 2, 1),
        #                            Mish())
        self.conv5 = nn.Conv3D(in_channels * 4, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias_attr=False)

        self.conv6 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())
        self.conv7 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 4, in_channels * 4, 3, padding=1, output_padding=1, stride=2, bias_attr=False),
            nn.BatchNorm3D(in_channels * 4))

        self.conv8 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias_attr=False),
            nn.BatchNorm3D(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias_attr=False),
            nn.BatchNorm3D(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                      Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())
        self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)

    def forward(self, x, feature4, feature5, feature6):
        conv1 = self.conv1(x)  # 1/8
        conv1 = paddle.concat((conv1, feature4), axis=1)  # 1/8
        conv1 = self.combine1(conv1)  # 1/8
        conv2 = self.conv2(conv1)  # 1/8

        conv3 = self.conv3(conv2)  # 1/16
        conv3 = paddle.concat((conv3, feature5), axis=1)  # 1/16
        conv3 = self.combine2(conv3)  # 1/16
        conv4 = self.conv4(conv3)  # 1/16

        conv5 = self.conv5(conv4)  # 1/32
        conv5 = paddle.concat((conv5, feature6), axis=1)  # 1/32
        conv5 = self.combine3(conv5)
        conv6 = self.conv6(conv5)  # 1/32

        # conv7 = F.relu(self.conv7(conv6) + self.redir3(conv4), inplace=True)
        # conv8 = F.relu(self.conv8(conv7) + self.redir2(conv2), inplace=True)
        # conv9 = F.relu(self.conv9(conv8) + self.redir1(x), inplace=True)

        conv7 = FMish((self.conv7(conv6) + self.redir3(conv4)))
        conv8 = FMish(self.conv8(conv7) + self.redir2(conv2))
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))

        return conv9


class hourglass(nn.Layer):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias_attr=False),
            nn.BatchNorm3D(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.Conv3DTranspose(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias_attr=False),
            nn.BatchNorm3D(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6


class refinenet_version3(nn.Layer):
    def __init__(self, in_channels):
        super(refinenet_version3, self).__init__()

        self.inplanes = 128
        self.conv1 = nn.Sequential(
            convbn(in_channels, 128, 3, 1, 1, 1),
            Mish())

        # self.conv2 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
        # self.conv3 = self._make_layer(BasicBlock, 128, 1, 1, 1, 2)
        # self.conv4 = self._make_layer(BasicBlock, 128, 1, 1, 1, 4)
        self.conv2 = nn.Sequential(
            convbn(128, 128, 3, 1, 1, 1),
            Mish())
        self.conv3 = nn.Sequential(
            convbn(128, 128, 3, 1, 2, 2),
            Mish())
        self.conv4 = nn.Sequential(
            convbn(128, 128, 3, 1, 4, 4),
            Mish())
        self.conv5 = self._make_layer(BasicBlock, 96, 1, 1, 1, 8)
        self.conv6 = self._make_layer(BasicBlock, 64, 1, 1, 1, 16)
        self.conv7 = self._make_layer(BasicBlock, 32, 1, 1, 1, 1)

        self.conv8 = nn.Conv2D(32, 1, kernel_size=3, padding=1, stride=1, bias_attr=False)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x, disp):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        disp = disp + conv8

        return disp


class PWCNet(nn.Layer):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(PWCNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.combine1 = hourglassup(32)

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3D(32, 1, kernel_size=3, padding=1, stride=1, bias_attr=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3D(32, 1, kernel_size=3, padding=1, stride=1, bias_attr=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3D(32, 1, kernel_size=3, padding=1, stride=1, bias_attr=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3D(32, 1, kernel_size=3, padding=1, stride=1, bias_attr=False))

        self.classif4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3D(32, 1, kernel_size=3, padding=1, stride=1, bias_attr=False))

        self.refinenet3 = refinenet_version3(146)
        self.dispupsample = nn.Sequential(convbn(1, 32, 1, 1, 0, 1),
                                          Mish())

        for m in self.parameters():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume1 = build_gwc_volume(features_left["gw1"], features_right["gw1"], self.maxdisp // 4,
                                       self.num_groups)

        gwc_volume2 = build_gwc_volume(features_left["gw2"], features_right["gw2"], self.maxdisp // 8,
                                       self.num_groups)

        gwc_volume3 = build_gwc_volume(features_left["gw3"], features_right["gw3"], self.maxdisp // 16,
                                       self.num_groups)

        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 32,
                                       self.num_groups)
        if self.use_concat_volume:
            concat_volume1 = build_concat_volume(features_left["concat_feature1"], features_right["concat_feature1"],
                                                 self.maxdisp // 4)
            concat_volume2 = build_concat_volume(features_left["concat_feature2"], features_right["concat_feature2"],
                                                 self.maxdisp // 8)
            concat_volume3 = build_concat_volume(features_left["concat_feature3"], features_right["concat_feature3"],
                                                 self.maxdisp // 16)
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"],
                                                 self.maxdisp // 32)
            volume1 = paddle.concat((gwc_volume1, concat_volume1), 1)
            volume2 = paddle.concat((gwc_volume2, concat_volume2), 1)
            volume3 = paddle.concat((gwc_volume3, concat_volume3), 1)
            volume4 = paddle.concat((gwc_volume4, concat_volume4), 1)
        else:
            volume1 = gwc_volume1
            volume2 = gwc_volume2
            volume3 = gwc_volume3
            volume4 = gwc_volume4

        cost0 = self.dres0(volume1)
        cost0 = self.dres1(cost0) + cost0

        combine = self.combine1(cost0, volume2, volume3, volume4)
        # combine = self.combine1(cost0)
        out1 = self.dres2(combine)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)
        upsample_out = paddle.nn.Upsample(size=[self.maxdisp, left.shape[2],
                                                left.shape[3]], data_format='NCDHW', mode='trilinear')
        upsample_out_2d = paddle.nn.Upsample(size=[left.shape[2],
                                                   left.shape[3]], mode='bilinear')
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            # refinement = self.dia6(self.dia5(self.dia4(self.dia3(self.dia2(self.dia1(out3))))))
            # cost3 = cost3 + refinement
            combine = self.classif4(combine)
            cost0 = upsample_out(cost0)
            cost0 = layers.squeeze(input=cost0, axes=[1])
            pred0 = fluid.layers.softmax(cost0, axis=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = upsample_out(cost1)
            cost1 = layers.squeeze(input=cost1, axes=[1])
            pred1 = fluid.layers.softmax(cost1, axis=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = upsample_out(cost2)
            cost2 = layers.squeeze(input=cost2, axes=[1])
            pred2 = fluid.layers.softmax(cost2, axis=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = upsample_out(cost3)
            cost3 = layers.squeeze(input=cost3, axes=[1])
            pred3 = fluid.layers.softmax(cost3, axis=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            combine = upsample_out(combine)
            combine = layers.squeeze(input=combine, axes=[1])
            combine = fluid.layers.softmax(combine, axis=1)
            combine = disparity_regression(combine, self.maxdisp)

            pred3 = paddle.unsqueeze(pred3, 1)
            refinenet_feature_left = features_left["finetune_feature"]
            refinenet_feature_left = upsample_out_2d(refinenet_feature_left)
            refinenet_feature_right = features_right["finetune_feature"]
            refinenet_feature_right = upsample_out_2d(refinenet_feature_right)
            refinenet_feature_right_warp = warp(refinenet_feature_right, pred3)
            refinenet_costvolume = build_corrleation_volume(refinenet_feature_left, refinenet_feature_right_warp, 24, 1)
            refinenet_costvolume = layers.squeeze(input=refinenet_costvolume, axes=[1])

            pred3feature = self.dispupsample(pred3)
            refinenet_combine = paddle.concat((refinenet_feature_left - refinenet_feature_right_warp,
                                               refinenet_feature_left, pred3feature, pred3, refinenet_costvolume), axis = 1)
            disp_finetune = self.refinenet3(refinenet_combine, pred3)
            disp_finetune = layers.squeeze(input=disp_finetune, axes=[1])
            pred3 = paddle.squeeze(pred3, 1)
            # pred3 = torch.unsqueeze(pred3, 1)
            # pred3feature = self.dispupsample(pred3)
            # refinenet_combine = torch.cat((refinenet_feature, pred3feature, left), dim = 1)
            # disp_finetune = self.refinenet3(refinenet_combine, pred3)
            # disp_finetune = torch.squeeze(disp_finetune, 1)
            # pred3 = torch.squeeze(pred3, 1)
            return [pred0, combine, pred1, pred2, pred3, disp_finetune]

        else:
            cost3 = self.classif3(out3)
            # refinement = self.dia6(self.dia5(self.dia4(self.dia3(self.dia2(self.dia1(out3))))))
            # cost3 = cost3 + refinement
            cost3 = upsample_out(cost3)
            cost3 = layers.squeeze(input=cost3, axes=[1])
            pred3 = fluid.layers.softmax(cost3, axis=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            pred3 = paddle.unsqueeze(pred3, 1)
            refinenet_feature_left = features_left["finetune_feature"]
            refinenet_feature_left = upsample_out_2d(refinenet_feature_left)
            refinenet_feature_right = features_right["finetune_feature"]
            refinenet_feature_right = upsample_out_2d(refinenet_feature_right)
            refinenet_feature_right_warp = warp(refinenet_feature_right, pred3)
            refinenet_costvolume = build_corrleation_volume(refinenet_feature_left, refinenet_feature_right_warp, 24, 1)
            refinenet_costvolume = layers.squeeze(input=refinenet_costvolume, axes=[1])
            pred3feature = self.dispupsample(pred3)
            refinenet_combine = paddle.concat((
                refinenet_feature_left - refinenet_feature_right_warp, refinenet_feature_left,
                pred3feature, pred3, refinenet_costvolume), axis=1)
            disp_finetune = self.refinenet3(refinenet_combine, pred3)
            disp_finetune = layers.squeeze(input=disp_finetune, axes=[1])
            pred3 = layers.squeeze(input=pred3, axes=[1])

            return [pred3, disp_finetune]


def PWCNet_G(d):
    return PWCNet(d, use_concat_volume=False)


def PWCNet_GC(d):
    return PWCNet(d, use_concat_volume=True)
