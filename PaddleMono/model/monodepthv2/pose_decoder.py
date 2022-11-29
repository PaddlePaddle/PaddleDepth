import paddle
import paddle.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Layer):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2D(self.num_ch_enc[-1], 256, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[("pose", 0)] = nn.Conv2D(num_input_features * 256, 256, 3, stride, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[("pose", 1)] = nn.Conv2D(256, 256, 3, stride, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[("pose", 2)] = nn.Conv2D(256, 6 * num_frames_to_predict_for, 1, weight_attr=nn.initializer.KaimingUniform())

        self.relu = nn.ReLU()

        self.net = nn.LayerList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = paddle.concat(cat_features, axis=1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.reshape((-1, self.num_frames_to_predict_for, 1, 6))

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
