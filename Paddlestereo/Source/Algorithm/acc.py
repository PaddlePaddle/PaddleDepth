# -*- coding: utf-8 -*-
import paddle


def d_1(res: paddle.tensor, gt: paddle.tensor, start_threshold: int = 2,
        threshold_num: int = 4, related_error: float = 0.05,
        invalid_value: int = 0, max_disp: int = 192) -> paddle.tensor:
    mask = (gt != invalid_value) & (gt < max_disp)
    acc_res = []
    with paddle.no_grad():
        total_num = mask.sum()
        error = paddle.abs(res - gt) * mask
        related_threshold = gt * related_error * mask
        for i in range(threshold_num):
            threshold = start_threshold + i
            acc = (error > threshold) & (error > related_threshold)
            acc_num = acc.sum()
            error_rate = acc_num / (total_num + 1e-10)
            acc_res.append(error_rate)
        mae = error.sum() / (total_num + 1e-10)
    return acc_res, mae
