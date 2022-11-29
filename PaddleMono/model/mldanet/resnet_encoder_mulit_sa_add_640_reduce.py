# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.vision.models as models
from paddle.nn import functional as F
from paddle.utils.download import get_weights_path_from_url
from utils import load_weight_file


def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2D):
        if type == 'xavier':
            paddle.nn.initializer.XavierNormal(m.weight)
        elif type == 'kaiming':  # msra
            paddle.nn.initializer.KaimingNormal(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))


class ResNetMultiImageInput(paddle.vision.models.ResNet):
    def __init__(self, block, depth, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, depth)
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2D(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3 , bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                nn.initializer.KaimingNormal()(layer.weight)
            elif isinstance(layer, nn.BatchNorm2D):
                nn.initializer.Constant(1)(layer.weight)
                nn.initializer.Constant(0)(layer.bias)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 34, 50, 101, 152], "Can only run with 18, 34 50, 101, 152 layer resnet"
    block_type = models.resnet.BasicBlock if num_layers <= 34 else models.resnet.BottleneckBlock
    model = ResNetMultiImageInput(block_type, num_layers, num_input_images=num_input_images)

    if pretrained is True:
        loaded = paddle.load(get_weights_path_from_url(*models.resnet.model_urls['resnet{}'.format(num_layers)]))
        loaded['conv1.weight'] = paddle.concat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_dict(loaded)
    elif isinstance(pretrained, str):
        loaded = load_weight_file(pretrained)
        loaded['conv1.weight'] = paddle.concat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_dict(loaded)

    return model


class Attention_net(nn.Layer):
    def __init__(self, in_c, out_c):
        super(Attention_net, self).__init__()
        self.K = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.Q = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.V = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.local_weight = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias_attr=False)

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
        out = paddle.add(O, x)
        out = self.local_weight(out)
        return out


class ResnetEncoder_multi_sa_add_reduce_640(nn.Layer):
    """module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder_multi_sa_add_reduce_640, self).__init__()

        self.pool = nn.MaxPool2D(2, 2)

        self.anet_512 = Attention_net(512, 512)
        self.anet_128 = Attention_net(128, 128)
        self.anet_64_1 = Attention_net(64, 64)
        self.anet_64_2 = Attention_net(64, 64)
        self.anet_256 = Attention_net(256, 256)

        self.convs10 = nn.Conv2D(3 * num_input_images, 64, 3, 1, 1)
        self.bn10 = nn.BatchNorm2D(64)
        self.convs11 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn11 = nn.BatchNorm2D(64)

        self.convs20 = nn.Conv2D(3 * num_input_images, 64, 3, 1, 1)
        self.bn20 = nn.BatchNorm2D(64)
        self.convs21 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn21 = nn.BatchNorm2D(64)

        self.convs30 = nn.Conv2D(3 * num_input_images, 128, 3, 1, 1)
        self.bn30 = nn.BatchNorm2D(128)
        self.convs31 = nn.Conv2D(128, 128, 3, 1, 1)
        self.bn31 = nn.BatchNorm2D(128)

        self.convs40 = nn.Conv2D(3 * num_input_images, 256, 3, 1, 1)
        self.bn40 = nn.BatchNorm2D(256)
        self.convs41 = nn.Conv2D(256, 256, 3, 1, 1)
        self.bn41 = nn.BatchNorm2D(256)

        self.convs50 = nn.Conv2D(3 * num_input_images, 512, 3, 1, 1)
        self.bn50 = nn.BatchNorm2D(512)
        self.convs51 = nn.Conv2D(512, 512, 3, 1, 1)
        self.bn51 = nn.BatchNorm2D(512)

        self.relu = nn.ReLU()

        self.convs64_1 = nn.Conv2D(128, 64, 3, 1, 1)
        self.bn64_1 = nn.BatchNorm2D(64)
        self.convs64_2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn64_2 = nn.BatchNorm2D(64)

        self.convs64_3 = nn.Conv2D(128, 64, 3, 1, 1)
        self.bn64_3 = nn.BatchNorm2D(64)
        self.convs64_4 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn64_4 = nn.BatchNorm2D(64)

        self.convs128_1 = nn.Conv2D(256, 128, 3, 1, 1)
        self.bn128_1 = nn.BatchNorm2D(128)
        self.convs128_2 = nn.Conv2D(128, 128, 3, 1, 1)
        self.bn128_2 = nn.BatchNorm2D(128)

        self.convs256_1 = nn.Conv2D(512, 256, 3, 1, 1)
        self.bn256_1 = nn.BatchNorm2D(256)
        self.convs256_2 = nn.Conv2D(256, 256, 3, 1, 1)
        self.bn256_2 = nn.BatchNorm2D(256)

        self.convs512_1 = nn.Conv2D(1024, 512, 3, 1, 1)
        self.bn512_1 = nn.BatchNorm2D(512)
        self.convs512_2 = nn.Conv2D(512, 512, 3, 1, 1)
        self.bn512_2 = nn.BatchNorm2D(512)

        self.upsample1 = nn.UpsamplingBilinear2D(size=(96, 320))
        self.upsample2 = nn.UpsamplingBilinear2D(size=(48, 160))
        self.upsample3 = nn.UpsamplingBilinear2D(size=(24, 80))
        self.upsample4 = nn.UpsamplingBilinear2D(size=(12, 40))
        self.upsample5 = nn.UpsamplingBilinear2D(size=(6, 20))

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if pretrained == 'pretrained':
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, True, num_input_images)
            else:
                self.encoder = resnets[num_layers](True)
        elif pretrained == 'scratch':
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, False, num_input_images)
            else:
                self.encoder = resnets[num_layers](False)
        else:
            if num_input_images > 1:
                self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
            else:
                self.encoder = resnets[num_layers](False)
                self.encoder.load_dict(load_weight_file(pretrained))

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):

        self.features = []
        x = (input_image - 0.45) / 0.225
        x_tmp = x

        x_tmp1 = self.upsample1(x_tmp)
        x_tmp2 = self.upsample2(x_tmp1)
        x_tmp3 = self.upsample3(x_tmp2)
        x_tmp4 = self.upsample4(x_tmp3)
        x_tmp5 = self.upsample5(x_tmp4)

        # (96, 320)
        x_tmp10 = self.relu(self.bn10(self.convs10(x_tmp1)))

        x_tmp11 = self.relu(self.bn11(self.convs11(x_tmp10)))

        # (48, 160)
        x_tmp20 = self.relu(self.bn20(self.convs20(x_tmp2)))
        x_tmp21 = self.relu(self.bn21(self.convs21(x_tmp20)))

        # (24, 80)
        x_tmp30 = self.relu(self.bn30(self.convs30(x_tmp3)))
        x_tmp31 = self.relu(self.bn31(self.convs31(x_tmp30)))

        # (12, 40)
        x_tmp40 = self.relu(self.bn40(self.convs40(x_tmp4)))
        x_tmp41 = self.relu(self.bn41(self.convs41(x_tmp40)))

        # (6, 20)
        x_tmp50 = self.relu(self.bn50(self.convs50(x_tmp5)))
        x_tmp51 = self.relu(self.bn51(self.convs51(x_tmp50)))

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        x_feature = self.encoder.relu(x)

        x_tmp = paddle.concat((x_feature, x_tmp11), 1)
        x_tmp = self.relu(self.bn64_1(self.convs64_1(x_tmp)))
        x_tmp = self.relu(self.bn64_2(self.convs64_2(x_tmp)))
        x_tmp = self.anet_64_1(x_tmp)
        self.features.append(x_tmp)

        x_feature = self.encoder.layer1(self.encoder.maxpool(x_feature))
        x_tmp = paddle.concat((x_feature, x_tmp21), 1)
        x_tmp = self.relu(self.bn64_3(self.convs64_3(x_tmp)))
        x_tmp = self.relu(self.bn64_4(self.convs64_4(x_tmp)))
        x_tmp = self.anet_64_2(x_tmp)
        self.features.append(x_tmp)

        x_feature = self.encoder.layer2(x_feature)
        x_tmp = paddle.concat((x_feature, x_tmp31), 1)
        x_tmp = self.relu(self.bn128_1(self.convs128_1(x_tmp)))
        x_tmp = self.relu(self.bn128_2(self.convs128_2(x_tmp)))
        x_tmp = self.anet_128(x_tmp)
        self.features.append(x_tmp)

        x_feature = self.encoder.layer3(x_feature)
        x_tmp = paddle.concat((x_feature, x_tmp41), 1)
        x_tmp = self.relu(self.bn256_1(self.convs256_1(x_tmp)))
        x_tmp = self.relu(self.bn256_2(self.convs256_2(x_tmp)))
        x_tmp = self.anet_256(x_tmp)
        self.features.append(x_tmp)

        x_feature = self.encoder.layer4(x_feature)
        x_tmp = paddle.concat((x_feature, x_tmp51), 1)
        x_tmp = self.relu(self.bn512_1(self.convs512_1(x_tmp)))
        x_tmp = self.relu(self.bn512_2(self.convs512_2(x_tmp)))
        x_tmp = self.anet_512(x_tmp)
        self.features.append(x_tmp)

        return self.features
