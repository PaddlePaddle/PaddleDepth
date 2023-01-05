# -*- coding: utf-8 -*-
from paddle.optimizer.lr import LRScheduler


class StereoLRScheduler(LRScheduler):
    def __init__(self, lr: float, stage: list,
                 last_epoch: int = -1, verbose: bool = False):
        self.stage = stage
        super().__init__(lr, last_epoch, verbose)

    def get_lr(self):
        new_lr = self.base_lr
        for item in self.stage:
            if item <= self.last_epoch:
                new_lr = new_lr * 0.1
            else:
                return new_lr
        return new_lr
