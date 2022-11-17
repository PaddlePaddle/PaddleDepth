import math

import paddle
import paddle.nn as nn


class DenseBlock(nn.Layer):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias_attr=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm1D(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act =nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(nn.Layer):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, activation='prelu', norm=None, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2D(input_size, output_size, kernel_size, stride, padding, bias_attr=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn =nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Layer):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation='prelu', norm=None,  bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = nn.Conv2DTranspose(input_size, output_size, kernel_size, stride, padding, bias_attr=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

        
class FeedbackBlock1(nn.Layer):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.avgpool_1 = nn.AvgPool2D(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation=activation, norm=None)
        self.act_1 = nn.ReLU()

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class FeedbackBlock2(nn.Layer):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = nn.AvgPool2D(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = nn.ReLU()

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0

class channel_attentionBlock(nn.Layer):
    def __init__(self, num_filter):
        super(channel_attentionBlock, self).__init__()

        self.g_aver_pooling1 = nn.AdaptiveAvgPool2D(1)

        self.fc1 = nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = nn.Sigmoid()   

    def forward(self, x): 
        x1 = self.g_aver_pooling1(x)
        x1 = x1.reshape([x1.shape[0], -1])
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.reshape([act2.shape[0], act2.shape[1], 1, 1])

        y = x + x*act2

        return y


class DilaConvBlock(nn.Layer):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=2, bias=True, activation='prelu', norm=None):
        super(DilaConvBlock, self).__init__()
        self.conv = nn.Conv2D(input_size, output_size, kernel_size, stride, padding,  dilation, bias_attr=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class MultiViewBlock1(nn.Layer):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock1, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)   

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = paddle.concat([x, x_prior1], axis=1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = paddle.concat([concat1, x_prior2], axis=1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = paddle.concat([concat2, x_prior3], axis=1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = paddle.concat([concat3, x_prior4], axis=1)

        x_prior1_2 = self.dilaconv1_2(concat_p1)
        
        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class MultiViewBlock2(nn.Layer):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock2, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = paddle.concat([x, x_prior1],1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = paddle.concat([concat1, x_prior2],1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = paddle.concat([concat2, x_prior3],1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = paddle.concat([concat3, x_prior4],1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class MultiViewBlock3(nn.Layer):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock3, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(4*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(5*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(6*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(7*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)  

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = paddle.concat([x, x_prior1],1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = paddle.concat([concat1, x_prior2],1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = paddle.concat([concat2, x_prior3],1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = paddle.concat([concat3, x_prior4],1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class MultiViewBlock4(nn.Layer):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock4, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(5*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(6*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(7*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(8*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)  

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = paddle.concat([x, x_prior1],1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = paddle.concat([concat1, x_prior2],1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = paddle.concat([concat2, x_prior3],1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = paddle.concat([concat3, x_prior4],1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class MultiViewBlock5(nn.Layer):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock5, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(6*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(7*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(8*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(9*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)    

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = paddle.concat([x, x_prior1],1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = paddle.concat([concat1, x_prior2],1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = paddle.concat([concat2, x_prior3],1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = paddle.concat([concat3, x_prior4],1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1