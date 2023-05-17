import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .update import BasicMultiUpdateBlock
from .extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from .corr import CorrBlock1D, PaddleAlternateCorrBlock1D
from .corr import coords_grid, upflow8


class RAFTStereo(nn.Layer):
    def __init__(self, 
                 n_downsample=2, 
                 mixed_precision=False,
                 corr_implementation="reg",
                 corr_levels=4, 
                 corr_radius=4, 
                 n_gru_layers=3, 
                 hidden_dims=[128]*3, 
                 shared_backbone=False, 
                 slow_fast_gru=False):
        super().__init__()       
        context_dims = hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[hidden_dims, context_dims], norm_fn="batch", downsample=n_downsample)
        self.update_block = BasicMultiUpdateBlock(corr_levels, corr_radius, n_gru_layers, n_downsample, hidden_dims=hidden_dims)

        self.context_zqr_convs = nn.LayerList([nn.Conv2D(context_dims[i], hidden_dims[i]*3, 3, padding=3//2) for i in range(n_gru_layers)])

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.shared_backbone = shared_backbone
        self.n_downsample = n_downsample
        self.mixed_precision = mixed_precision
        self.n_gru_layers = n_gru_layers
        self.corr_implementation = corr_implementation
        self.slow_fast_gru = slow_fast_gru

        if shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2D(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=n_downsample)

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, nn.BatchNorm2D):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W)
        coords1 = coords_grid(N, H, W)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.n_downsample
        mask = mask.reshape([N, 1, 9, factor, factor, H, W])
        mask = F.softmax(mask, axis=2)

        up_flow = F.unfold(factor * flow, [3,3], paddings=1)
        up_flow = up_flow.reshape([N, D, 9, 1, 1, H, W])

        up_flow = paddle.sum(mask * up_flow, axis=2)
        up_flow = up_flow.transpose([0, 1, 4, 2, 5, 3])
        return up_flow.reshape([N, D, factor*H, factor*W])


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0)
        image2 = (2 * (image2 / 255.0) - 1.0)

        # run the context network
        with paddle.amp.auto_cast(enable=self.mixed_precision):
            if self.shared_backbone:
                *cnet_list, x = self.cnet(paddle.concat((image1, image2), axis=0), dual_inp=True, num_layers=self.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(axis=0, num_or_sections=2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [paddle.tanh(x[0]) for x in cnet_list]
            inp_list = [F.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(num_or_sections=3, axis=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.astype('float32'), fmap2.astype('float32')
        elif self.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PaddleAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.astype('float32'), fmap2.astype('float32')

        corr_fn = corr_block(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with paddle.amp.auto_cast(enable=self.mixed_precision):
                if self.n_gru_layers == 3 and self.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.n_gru_layers >= 2 and self.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.n_gru_layers==3, iter16=self.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
