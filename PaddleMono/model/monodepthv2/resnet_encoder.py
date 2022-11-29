import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.models as models
from paddle.utils.download import get_weights_path_from_url
from utils import load_weight_file

class ResNetMultiImageInput(models.ResNet):
    """
    Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/models/resnet.py
    """
    def __init__(self, block, depth, num_classes=1000, with_pool=True, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, depth, num_classes, with_pool)
        self.conv1 = nn.Conv2D(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, weight_attr=nn.initializer.KaimingUniform(), bias_attr=False) # replace the first conv


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """
    Constructs a ResNet model with multiple input images.
    Args:
        num_layers (int): Number of resnet layers. Must be 18, 34 50, 101, 152
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


class ResnetEncoder(nn.Layer):
    """
    Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

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
        x = (input_image - 0.45) / 0.225 # normalization?
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
