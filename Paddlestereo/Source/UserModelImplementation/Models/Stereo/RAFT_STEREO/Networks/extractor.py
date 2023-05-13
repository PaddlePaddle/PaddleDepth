import paddle
import paddle.nn as nn
from .initializer import kaiming_normal_, constant_


class ResidualBlock(nn.Layer):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2D(planes)
            self.norm2 = nn.BatchNorm2D(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2D(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2D(planes, weight_attr=False, bias_attr=False)
            self.norm2 = nn.InstanceNorm2D(planes, weight_attr=False, bias_attr=False)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2D(planes, weight_attr=False, bias_attr=False)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2D(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Layer):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2D(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2D(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2D(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2D(planes//4)
            self.norm2 = nn.BatchNorm2D(planes//4)
            self.norm3 = nn.BatchNorm2D(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2D(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2D(planes//4, weight_attr=False, bias_attr=False)
            self.norm2 = nn.InstanceNorm2D(planes//4, weight_attr=False, bias_attr=False)
            self.norm3 = nn.InstanceNorm2D(planes, weight_attr=False, bias_attr=False)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2D(planes, weight_attr=False, bias_attr=False)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2D(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Layer):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2D(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2D(64, weight_attr=False, bias_attr=False)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU()

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # output convolution
        self.conv2 = nn.Conv2D(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2D(p=dropout)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                if m.weight is not None:
                    constant_(m.weight, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2D):
                if m.scale is not None:
                    constant_(m.scale, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = paddle.concat(x, axis=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            split_num = x.shape[0]//batch_dim
            x = x.split(num_or_sections=split_num, axis=0)

        return x

class MultiBasicEncoder(nn.Layer):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2D(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2D(64, weight_attr=False, bias_attr=False)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU()

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2D(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.LayerList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2D(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.LayerList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2D(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.LayerList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2D(p=dropout)
        else:
            self.dropout = None

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                if m.weight is not None:
                    constant_(m.weight, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2D):
                if m.scale is not None:
                    constant_(m.scale, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)
