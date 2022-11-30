import paddle.nn as nn
import paddle


class PhotometricLoss(nn.Layer):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, target, recon, mask=None):

        assert recon.dim(
        ) == 4, "expected recon dimension to be 4, but instead got {}.".format(
            recon.dim())
        assert target.dim(
        ) == 4, "expected target dimension to be 4, but instead got {}.".format(
            target.dim())
        assert recon.size() == target.size(), "expected recon and target to have the same size, but got {} and {} instead" \
            .format(recon.size(), target.size())
        diff = (target - recon).abs()
        diff = paddle.sum(diff, 1)  # sum along the color channel

        # compare only pixels that are not black
        valid_mask = (paddle.sum(recon, 1) > 0).float() * (paddle.sum(target, 1)
                                                           > 0).float()
        if mask is not None:
            valid_mask = valid_mask * paddle.squeeze(mask).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                print(
                    "warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size)."
                )
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss
