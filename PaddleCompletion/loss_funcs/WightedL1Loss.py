import paddle


class WightedL1Loss(paddle.nn.Layer):
    def __init__(self):
        super(WightedL1Loss, self).__init__()

    def forward(self, pred, label):
        label_mask = label > 0.0001
        _pred = pred[label_mask]
        _label = label[label_mask]
        n_valid_element = _label.shape[0]
        diff_mat = paddle.abs(_pred - _label)
        loss = paddle.sum(diff_mat) / n_valid_element
        return loss
