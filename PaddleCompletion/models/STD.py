import paddle

import paddle.nn as nn

from paddle.vision.models import resnet

import paddle.nn.functional as F



def init_weights(m):

    init = nn.initializer.Normal(mean=0, std=1e-3)

    zeros = nn.initializer.Constant(0.)

    ones = nn.initializer.Constant(1.)

    if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):

        # m.weight.data.normal_(0, 1e-3)

        init(m.weight)

        if m.bias is not None:

            # m.bias.data.zero_()

            zeros(m.bias)

    elif isinstance(m, nn.Conv2DTranspose):

        # m.weight.data.normal_(0, 1e-3)

        init(m.weight)

        if m.bias is not None:

            # m.bias.data.zero_()

            zeros(m.bias)

    elif isinstance(m, nn.BatchNorm2D):

        # m.weight.data.fill_(1) torch

        # m.bias.data.zero_() torch

        ones(m.weight)

        zeros(m.weight)





def conv_bn_relu(in_channels, out_channels, kernel_size, \

                 stride=1, padding=0, bn=True, relu=True):

    bias_attr = not bn

    layers = []

    layers.append(

        nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=bias_attr))

    if bn:

        layers.append(nn.BatchNorm2D(out_channels))

    if relu:

        layers.append(nn.LeakyReLU(0.2))

    layers = nn.Sequential(*layers)

    for m in layers.sublayers():

        init_weights(m)



    return layers





def convt_bn_relu(in_channels, out_channels, kernel_size, \

                  stride=1, padding=0, output_padding=0, bn=True, relu=True):

    bias_attr = not bn

    layers = []

    layers.append(

        nn.Conv2DTranspose(in_channels,

                           out_channels,

                           kernel_size,

                           stride,

                           padding,

                           output_padding, bias_attr=bias_attr))

    if bn:

        layers.append(nn.BatchNorm2D(out_channels))

    if relu:

        layers.append(nn.LeakyReLU(0.2))

    layers = nn.Sequential(*layers)

    for m in layers.sublayers():

        init_weights(m)



    return layers





class DepthCompletionSTDNet(nn.Layer):

    def __init__(self,args):

        assert args.layers in [18, 34, 50, 101,

                               152], 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(args.layers)

        super(DepthCompletionSTDNet,self).__init__()

        

        self.modality = args.dataset["input_mode"]  # input type

        # print(len(self.modality))

        self.layers = args.layers

        if 'd' in self.modality:

            channels = 64 // len(self.modality)

            #   self.conv1_d = conv_bn_relu(1,channels,kernel_size=3,stride=1,padding=1)

            self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        if 'rgb' in self.modality:

            channels = 64 * 3 // len(self.modality)

            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)

        elif 'g' in self.modality:

            channels = 64 // len(self.modality)

            # self.conv1_img = conv_bn_relu(1,channels,kernel_size=3,stride=1,padding=1)

            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)



        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)

        # print(pretrained_model)

        if not args.pretrained:

            pretrained_model.apply(init_weights)

        # self.maxpool = pretrained_model._modules['maxpool']

        self.conv2 = pretrained_model.layer1

        self.conv3 = pretrained_model.layer2

        self.conv4 = pretrained_model.layer3

        self.conv5 = pretrained_model.layer4

        del pretrained_model  # clear memory



        # define number of intermediate channels

        if args.layers <= 34:

            num_channels = 512

        elif args.layers >= 50:

            num_channels = 2048

        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers

        kernel_size = 3

        stride = 2

        self.convt5 = convt_bn_relu(in_channels=(512), out_channels=256, kernel_size=kernel_size,

                                    stride=stride, padding=1, output_padding=1)

        self.convt4 = convt_bn_relu(in_channels=(768 ), out_channels=128, kernel_size=kernel_size,

                                    stride=stride, padding=1, output_padding=1)

        self.convt3 = convt_bn_relu(in_channels=(256 + 128 ), out_channels=64, kernel_size=kernel_size,

                                    stride=stride, padding=1, output_padding=1)

        self.convt2 = convt_bn_relu(in_channels=(128 + 64 ), out_channels=64, kernel_size=kernel_size,

                                    stride=stride, padding=1, output_padding=1)

        self.convt1 = convt_bn_relu(in_channels=(128 ), out_channels=64, kernel_size=kernel_size, stride=1,

                                    padding=1)

        self.convtf = conv_bn_relu(in_channels=(128 ), out_channels=1, kernel_size=1, stride=1, bn=False,

                                   relu=False)

    def forward(self,x):

        # first layer

        if 'd' in self.modality:

            conv1_d = self.conv1_d(x['d'])

        if 'rgb' in self.modality:

            conv1_img = self.conv1_img(x['rgb'])

        elif 'g' in self.modality:

            conv1_img = self.conv1_img(x['g'])



        if self.modality == 'rgbd' or self.modality == 'gd':

            conv1 = paddle.concat((conv1_d, conv1_img), 1)

        else:

            conv1 = conv1_d if (self.modality == 'd') else conv1_img



        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608

        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304

        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152

        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76



        # decoder

        convt5 = self.convt5(conv6)

        y = paddle.concat((convt5, conv5), 1)



        convt4 = self.convt4(y)

        y = paddle.concat((convt4, conv4), 1)



        convt3 = self.convt3(y)

        y = paddle.concat((convt3, conv3), 1)



        convt2 = self.convt2(y)

        y = paddle.concat((convt2, conv2), 1)



        convt1 = self.convt1(y)

        y = paddle.concat((convt1, conv1), 1)



        y = self.convtf(y)



        if self.training:

            return 100 * y

        else:

            min_distance = 0.9

            return F.relu(

                100 * y - min_distance

            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m




