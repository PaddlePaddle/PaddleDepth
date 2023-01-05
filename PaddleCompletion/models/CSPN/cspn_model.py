import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .cspn import Affinity_Propagate


class BatchNorm2D(nn.BatchNorm2D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class UnPool(nn.Layer):
    def __init__(self, num_channels, stride=2):
        super().__init__()
        self.num_channels = num_channels
        self.stride = stride
        # create kernel [1, 0; 0, 0]
        kernel = paddle.zeros((num_channels, 1, stride, stride), dtype='float32')
        kernel[:, :, 0, 0] = 1
        self.weights = kernel

    def forward(self, x):
        return F.conv2d_transpose(x, self.weights, stride=self.stride, groups=self.num_channels)


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False),
            BatchNorm2D(planes),
            nn.ReLU(),
            nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False),
            BatchNorm2D(planes),
            nn.ReLU(),
            nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False),
            BatchNorm2D(planes * 4),
        )
        self.down_sample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        return self.relu(out + x)


class UpProj_Block(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias_attr=False),
            BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            BatchNorm2D(out_channels),
        )
        self.short_cut = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias_attr=False),
            BatchNorm2D(out_channels)
        )
        self.relu = nn.ReLU()
        self._up_pool = UnPool(in_channels)

    def _up_pooling(self, x):
        x = self._up_pool(x)
        return x

    def forward(self, x):
        x = self._up_pooling(x)
        out = self.block(x)
        sc = self.short_cut(x)
        return self.relu(out + sc)


class Simple_Gudi_UpConv_Block_Last_Layer(nn.Layer):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block_Last_Layer, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = UnPool(in_channels)

    def _up_pooling(self, x, scale):
        x = self._up_pool(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x[:, :, 0:self.oheight, 0:self.owidth]
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.conv1(x)
        return out


class Gudi_UpProj_Block(nn.Layer):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias_attr=False)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = BatchNorm2D(out_channels)
        self.sc_conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias_attr=False)
        self.sc_bn1 = BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.oheight = oheight
        self.owidth = owidth

    def _up_pooling(self, x, scale):
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x[:, :, 0:self.oheight, 0:self.owidth]
        mask = paddle.zeros_like(x)
        mask.stop_gradient = True
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:, :, h, w] = 1
        x = paddle.multiply(mask, x)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Gudi_UpProj_Block_Cat(nn.Layer):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block_Cat, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias_attr=False)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv1_1 = nn.Conv2D(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1_1 = BatchNorm2D(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = BatchNorm2D(out_channels)
        self.sc_conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias_attr=False)
        self.sc_bn1 = BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = UnPool(in_channels)

    def _up_pooling(self, x):
        x = self._up_pool(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x[:, :, 0:self.oheight, 0:self.owidth]
        return x

    def forward(self, x, side_input):
        x = self._up_pooling(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = paddle.concat((out, side_input), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(self, block, layers, up_proj_block, cspn_config=None):
        super().__init__()
        self.inplanes = 64
        cspn_config_default = {'step': 24, 'kernel': 3, 'norm_type': '8sum_abs'}
        if not (cspn_config is None):
            cspn_config_default.update(cspn_config)
        print(cspn_config_default)

        self.conv1_1 = nn.Conv2D(4, 64, kernel_size=7, stride=2, padding=3,
                                 bias_attr=False)
        self.bn1 = BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mid_channel = 256 * block.expansion
        self.conv2 = nn.Conv2D(512 * block.expansion, 512 * block.expansion, kernel_size=3,
                               stride=1, padding=1, bias_attr=False)
        self.bn2 = BatchNorm2D(512 * block.expansion)
        self.up_proj_layer1 = self._make_up_conv_layer(up_proj_block,
                                                       self.mid_channel,
                                                       int(self.mid_channel / 2))
        self.up_proj_layer2 = self._make_up_conv_layer(up_proj_block,
                                                       int(self.mid_channel / 2),
                                                       int(self.mid_channel / 4))
        self.up_proj_layer3 = self._make_up_conv_layer(up_proj_block,
                                                       int(self.mid_channel / 4),
                                                       int(self.mid_channel / 8))
        self.up_proj_layer4 = self._make_up_conv_layer(up_proj_block,
                                                       int(self.mid_channel / 8),
                                                       int(self.mid_channel / 16))
        self.conv3 = nn.Conv2D(128, 1, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.post_process_layer = self._make_post_process_layer(cspn_config_default)
        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(Gudi_UpProj_Block, 2048, 1024, 15, 19)
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 1024, 512, 29, 38)
        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 512, 256, 57, 76)
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 256, 64, 114, 152)
        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 1, 228, 304)
        self.gud_up_proj_layer6 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 8, 228, 304)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_up_conv_layer(self, up_proj_block, in_channels, out_channels):
        return up_proj_block(in_channels, out_channels)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth):
        return up_proj_block(in_channels, out_channels, oheight, owidth)

    def _make_post_process_layer(self, cspn_config=None):
        return Affinity_Propagate(cspn_config['step'],
                                  cspn_config['kernel'],
                                  norm_type=cspn_config['norm_type'])

    def forward(self, x):
        # batch_size, channel, height, width = x.size()
        sparse_depth = x.clone()[:, 3:4, :, :]
        x = self.conv1_1(x)
        skip4 = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skip3 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(self.conv2(x))
        x = self.gud_up_proj_layer1(x)
        x = self.gud_up_proj_layer2(x, skip2)
        x = self.gud_up_proj_layer3(x, skip3)
        x = self.gud_up_proj_layer4(x, skip4)

        guidance = self.gud_up_proj_layer6(x)
        x = self.gud_up_proj_layer5(x)

        x = self.post_process_layer(guidance, x, sparse_depth)
        return x


def UnetCSPN(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], UpProj_Block, {'norm_type': '8sum_abs'})

    if pretrained:
        model_dict = model.state_dict()
        from paddle.vision.models import resnet50 as resnet_pretrain
        pretrained_model = resnet_pretrain(pretrained=True)
        pretrained_model_dict = pretrained_model.state_dict()
        model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(model_dict)
        print('Load pretrained model from resnet50.')

    return model
