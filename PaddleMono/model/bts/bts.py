import paddle
import paddle.nn as nn
import math

nn.initializer.set_global_initializer(paddle.nn.initializer.XavierUniform(), nn.initializer.Constant())

class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_sublayer('first_bn', nn.BatchNorm2D(in_channels, momentum=0.01, epsilon=1.1e-5))
        
        self.atrous_conv.add_sublayer('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2D(in_channels=in_channels, out_channels=out_channels*2, bias_attr=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2D(out_channels*2, momentum=0.01),
                                                                    nn.ReLU(),
                                                                    nn.Conv2D(in_channels=out_channels * 2, out_channels=out_channels, bias_attr=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv(x)

class upconv(nn.Layer):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, bias_attr=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = nn.functional.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out

class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = paddle.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_sublayer('final', paddle.nn.Sequential(nn.Conv2D(num_in_filters, out_channels=1, bias_attr=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_sublayer('plane_params', paddle.nn.Conv2D(num_in_filters, out_channels=3, bias_attr=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_sublayer('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      paddle.nn.Sequential(nn.Conv2D(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias_attr=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = paddle.multiply(paddle.sin(theta), paddle.cos(phi)).unsqueeze(1)
            n2 = paddle.multiply(paddle.sin(theta), paddle.sin(phi)).unsqueeze(1)
            n3 = paddle.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = paddle.concat([n1, n2, n3, n4], axis=1)
        
        return net

class local_planar_guidance(nn.Layer):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = paddle.arange(self.upratio).reshape([1, 1, self.upratio]).astype('float32')
        self.v = paddle.arange(int(self.upratio)).reshape([1, self.upratio, 1]).astype('float32')
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = paddle.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = paddle.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.tile([plane_eq.shape[0], plane_eq.shape[2] * int(self.upratio), plane_eq.shape[3]])
        u = (u - (self.upratio - 1) * 0.5) / self.upratio
        
        v = self.v.tile([plane_eq.shape[0], plane_eq.shape[2], plane_eq.shape[3] * int(self.upratio)])
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)

class bts(nn.Layer):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(bts, self).__init__()
        self.params = params

        self.upconv5    = upconv(feat_out_channels[4], num_features)
        self.bn5        = nn.BatchNorm2D(num_features, momentum=0.01, epsilon=1.1e-5)
        
        self.conv5      = paddle.nn.Sequential(nn.Conv2D(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias_attr=False),
                                              nn.ELU())
        self.upconv4    = upconv(num_features, num_features // 2)
        self.bn4        = nn.BatchNorm2D(num_features // 2, momentum=0.01, epsilon=1.1e-5)
        self.conv4      = paddle.nn.Sequential(nn.Conv2D(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias_attr=False),
                                              nn.ELU())
        self.bn4_2      = nn.BatchNorm2D(num_features // 2, momentum=0.01, epsilon=1.1e-5)
        
        self.daspp_3    = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6    = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12   = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18   = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24   = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = paddle.nn.Sequential(nn.Conv2D(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias_attr=False),
                                              nn.ELU())
        self.reduc8x8   = reduction_1x1(num_features // 4, num_features // 4, self.params.max_depth)
        self.lpg8x8     = local_planar_guidance(8)
        
        self.upconv3    = upconv(num_features // 4, num_features // 4)
        self.bn3        = nn.BatchNorm2D(num_features // 4, momentum=0.01, epsilon=1.1e-5)
        self.conv3      = paddle.nn.Sequential(nn.Conv2D(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias_attr=False),
                                              nn.ELU())
        self.reduc4x4   = reduction_1x1(num_features // 4, num_features // 8, self.params.max_depth)
        self.lpg4x4     = local_planar_guidance(4)
        
        self.upconv2    = upconv(num_features // 4, num_features // 8)
        self.bn2        = nn.BatchNorm2D(num_features // 8, momentum=0.01, epsilon=1.1e-5)
        self.conv2      = paddle.nn.Sequential(nn.Conv2D(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias_attr=False),
                                              nn.ELU())
        
        self.reduc2x2   = reduction_1x1(num_features // 8, num_features // 16, self.params.max_depth)
        self.lpg2x2     = local_planar_guidance(2)
        
        self.upconv1    = upconv(num_features // 8, num_features // 16)
        self.reduc1x1   = reduction_1x1(num_features // 16, num_features // 32, self.params.max_depth, is_final=True)
        self.conv1      = paddle.nn.Sequential(nn.Conv2D(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias_attr=False),
                                              nn.ELU())
        self.get_depth  = paddle.nn.Sequential(nn.Conv2D(num_features // 16, 1, 3, 1, 1, bias_attr=False),
                                              nn.Sigmoid())

    def forward(self, features, focal):
        self.outputs = {}
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        dense_features = paddle.nn.ReLU()(features[4])
        upconv5 = self.upconv5(dense_features) # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = paddle.concat([upconv5, skip3], axis=1)
        iconv5 = self.conv5(concat5)
        
        upconv4 = self.upconv4(iconv5) # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = paddle.concat([upconv4, skip2], axis=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)
        
        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = paddle.concat([concat4, daspp_3], axis=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = paddle.concat([concat4_2, daspp_6], axis=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = paddle.concat([concat4_3, daspp_12], axis=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = paddle.concat([concat4_4, daspp_18], axis=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = paddle.concat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], axis=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        
        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = nn.functional.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = paddle.concat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.params.max_depth
        depth_8x8_scaled_ds = nn.functional.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')
        
        upconv3 = self.upconv3(daspp_feat) # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = paddle.concat([upconv3, skip1, depth_8x8_scaled_ds], axis=1)
        iconv3 = self.conv3(concat3)
        
        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = nn.functional.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = paddle.concat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.params.max_depth
        depth_4x4_scaled_ds = nn.functional.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')
        
        upconv2 = self.upconv2(iconv3) # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = paddle.concat([upconv2, skip0, depth_4x4_scaled_ds], axis=1)
        iconv2 = self.conv2(concat2)
        
        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = nn.functional.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = paddle.concat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.params.max_depth
        
        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = paddle.concat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], axis=1)
        iconv1 = self.conv1(concat1)
        final_depth = self.params.max_depth * self.get_depth(iconv1)
        if self.params.dataset == 'kitti':
            final_depth = final_depth * focal.reshape(-1, 1, 1, 1).float() / 715.0873

        self.outputs["depth_8x8_scaled"] = depth_8x8_scaled
        self.outputs["depth_4x4_scaled"] = depth_4x4_scaled
        self.outputs["depth_2x2_scaled"] = depth_2x2_scaled
        self.outputs["reduc1x1"] = reduc1x1
        self.outputs["final_depth"] = final_depth
        return self.outputs

class encoder(nn.Layer):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import paddle.vision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True, num_classes=0, with_pool=False)
            self.feat_names = ['conv1_func', 'pool2d_max', 'tr_conv2_blk', 'tr_conv3_blk', 'batch_norm']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True)
            self.feat_names = ['conv1_func', 'pool2d_max', 'tr_conv2_blk', 'tr_conv3_blk', 'batch_norm']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.base_model.named_children():
            if 'Linear' in k or 'fc' in k or 'classifier' in k or 'AvgPool' in k:
                continue
            if "out" not in k:
                feature = v(feature)
            if self.params.encoder == 'mobilenetv2_bts':
                if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                    skip_feat.append(feature)
            else:
                if any(x in k for x in self.feat_names):
                    skip_feat.append(feature)
            i = i + 1
        return skip_feat

class BtsModel(nn.Layer):
    def __init__(self, params):
        super(BtsModel, self).__init__()
        self.encoder = encoder(params)
        self.decoder = bts(params, self.encoder.feat_out_channels)

    def forward(self, x, focal):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)


