import paddle
import paddle.nn.functional as F


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], axis=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = paddle.concat([xgrid, ygrid], axis=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.astype('float32')

    return img

def coords_grid(batch, ht, wd):
    coords = paddle.meshgrid(paddle.arange(ht), paddle.arange(wd))
    coords = paddle.stack(coords[::-1], axis=0).astype('float32')
    return paddle.tile(coords[None], repeat_times=[batch, 1, 1, 1])


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class PaddleAlternateCorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], axis=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = paddle.concat([xgrid, ygrid], axis=-1)
        output_corr = []
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
            corr = paddle.sum(fmapw_mini * fmap1, axis=1)
            output_corr.append(corr)
        corr = paddle.stack(output_corr, axis=1).transpose([0,2,3,1])

        return corr /paddle.sqrt(paddle.to_tensor(D).astype('float32'))

    def __call__(self, coords):
        r = self.radius
        coords = coords.transpose([0, 2, 3, 1])
        batch, h1, w1, _ = coords.shape
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        out_pyramid = []
        for i in range(self.num_levels):
            dx = paddle.zeros([1])
            dy = paddle.linspace(-r, r, 2*r+1)
            delta = paddle.stack(paddle.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape([batch, h1, w1, 1, 2]).clone()
            centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i
            coords_lvl = centroid_lvl + delta.reshape([-1, 2])
            corr = self.corr(fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = paddle.concat(out_pyramid, axis=-1)
        return out.transpose([0, 3, 1, 2]).astype('float32')


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape([batch*h1*w1, 1, 1, w2])

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].transpose([0, 2, 3, 1])
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = paddle.linspace(-r, r, 2*r+1)
            dx = dx.reshape([2*r+1, 1])
            x0 = dx + coords.reshape([batch*h1*w1, 1, 1, 1]) / 2**i
            y0 = paddle.zeros_like(x0)

            coords_lvl = paddle.concat([x0,y0], axis=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.reshape([batch, h1, w1, -1])
            out_pyramid.append(corr)

        out = paddle.concat(out_pyramid, axis=-1)
        return out.transpose([0, 3, 1, 2]).astype('float32')

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.reshape([B, D, H, W1])
        fmap2 = fmap2.reshape([B, D, H, W2])
        corr = paddle.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape([B, H, W1, 1, W2])
        return corr / paddle.sqrt(paddle.to_tensor(D).astype('float32'))

