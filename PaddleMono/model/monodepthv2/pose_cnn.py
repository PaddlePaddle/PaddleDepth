import paddle
import paddle.nn as nn


class PoseCNN(nn.Layer):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2D(3 * num_input_frames, 16, 7, 2, 3, weight_attr=nn.initializer.KaimingUniform())
        self.convs[1] = nn.Conv2D(16, 32, 5, 2, 2, weight_attr=nn.initializer.KaimingUniform())
        self.convs[2] = nn.Conv2D(32, 64, 3, 2, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[3] = nn.Conv2D(64, 128, 3, 2, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[4] = nn.Conv2D(128, 256, 3, 2, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[5] = nn.Conv2D(256, 256, 3, 2, 1, weight_attr=nn.initializer.KaimingUniform())
        self.convs[6] = nn.Conv2D(256, 256, 3, 2, 1, weight_attr=nn.initializer.KaimingUniform())

        self.pose_conv = nn.Conv2D(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.LayerList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.reshape((-1, self.num_input_frames - 1, 1, 6))

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
