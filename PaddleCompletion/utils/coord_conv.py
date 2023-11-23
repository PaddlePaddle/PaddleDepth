import numpy as np


class AddCoordsNp:
    """Add coords to a tensor"""

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        xx_ones = np.ones([self.x_dim], dtype=np.int32)
        xx_ones = np.expand_dims(xx_ones, 1)
        xx_range = np.expand_dims(np.arange(self.y_dim), 0)
        xx_channel = np.matmul(xx_ones, xx_range)
        xx_channel = np.expand_dims(xx_channel, -1)
        yy_ones = np.ones([self.y_dim], dtype=np.int32)
        yy_ones = np.expand_dims(yy_ones, 0)
        yy_range = np.expand_dims(np.arange(self.x_dim), 1)
        yy_channel = np.matmul(yy_range, yy_ones)
        yy_channel = np.expand_dims(yy_channel, -1)
        xx_channel = xx_channel.astype("float32") / (self.y_dim - 1)
        yy_channel = yy_channel.astype("float32") / (self.x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        ret = np.concatenate([xx_channel, yy_channel], axis=-1)
        if self.with_r:
            rr = np.sqrt(np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
            ret = np.concatenate([ret, rr], axis=-1)
        return ret
