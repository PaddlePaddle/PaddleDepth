# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2


class MaskAug(object):
    """docstring for ClassName"""
    COLOR_GRAY = 127

    def __init__(self, img_height: int, img_width: int,
                 block_size: int = 2, ratio: float = 0.3) -> None:
        super().__init__()
        assert((img_height % block_size == 0) and (img_width % block_size == 0))
        self._img_height = img_height
        self._img_width = img_width
        self._block_size = block_size
        self._block_width_num = int(img_width / block_size)
        self._block_height_num = int(img_height / block_size)
        self._block_num = int(img_height * img_width / block_size / block_size)
        self._block_num_list = list(range(0, self._block_num))
        self._sample_num = int(self._block_num * ratio)

    def _generate_mask(self) -> np.array:
        random_sample_list = random.sample(self._block_num_list, self._sample_num)
        mask = np.ones([self._img_height, self._img_width], dtype = float)
        for sample_id in random_sample_list:
            height_id = sample_id // self._block_width_num
            width_id = sample_id % self._block_width_num
            cy = height_id * self._block_size
            cx = width_id * self._block_size
            mask[cy:cy + self._block_size, cx:cx + self._block_size] = 0
        return mask

    def __call__(self, img) -> np.array:
        mask = self._generate_mask()
        img[:, :, 0] = img[:, :, 0] * mask + (1 - mask) * MaskAug.COLOR_GRAY
        img[:, :, 1] = img[:, :, 1] * mask + (1 - mask) * MaskAug.COLOR_GRAY
        img[:, :, 2] = img[:, :, 2] * mask + (1 - mask) * MaskAug.COLOR_GRAY
        return img, mask


def debug_main():
    mask_aug = MaskAug(370, 1226, ratio=0.15)
    left_img = np.array(cv2.imread('./Example/000001_10_l.png'))
    left_img, mask = mask_aug(left_img)
    cv2.imwrite('./Example/000001_10_l_mask.png', left_img)


if __name__ == "__main__":
    debug_main()
