import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class FlowHead(nn.Layer):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2D(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2D(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Layer):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2D(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2D(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2D(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = paddle.concat(x_list, axis=1)
        hx = paddle.concat([h, x], axis=1)

        z = F.sigmoid(self.convz(hx) + cz)
        r = F.sigmoid(self.convr(hx) + cr)
        q = paddle.tanh(self.convq(paddle.concat([r*h, x], axis=1)) + cq)

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Layer):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2D(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = paddle.concat(x, axis=1)
        hx = paddle.concat([h, x], axis=1)
        z = F.sigmoid(self.convz1(hx))
        r = F.sigmoid(self.convr1(hx))
        q = paddle.tanh(self.convq1(paddle.concat([r*h, x], axis=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = paddle.concat([h, x], axis=1)
        z = F.sigmoid(self.convz2(hx))
        r = F.sigmoid(self.convr2(hx))
        q = paddle.tanh(self.convq2(paddle.concat([r*h, x], axis=1)))       
        h = (1-z) * h + z * q

        return h

class BasicMotionEncoder(nn.Layer):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()

        cor_planes = corr_levels * (2*corr_radius + 1)

        self.convc1 = nn.Conv2D(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2D(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2D(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2D(64, 64, 3, padding=1)
        self.conv = nn.Conv2D(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = paddle.concat([cor, flo], axis=1)
        out = F.relu(self.conv(cor_flo))
        return paddle.concat([out, flow], axis=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Layer):
    def __init__(self, corr_levels, corr_radius, n_gru_layers, n_downsample, hidden_dims=[]):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        encoder_output_dim = 128
        self.n_gru_layers = n_gru_layers

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**n_downsample

        self.mask = nn.Sequential(
            nn.Conv2D(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow
