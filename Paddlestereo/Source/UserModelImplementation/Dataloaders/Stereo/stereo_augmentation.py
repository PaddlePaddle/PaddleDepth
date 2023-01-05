# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageEnhance


class Augmentor(object):
    def __init__(self, image_height=384, image_width=512, max_disp=256,
                 scale_min=0.6, scale_max=1.0, seed=0,):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def chromatic_augmentation(img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part
        img_ = np.array(img)

        return img_

    def random_shift(self, left_img, right_img, left_disp):
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (self.rng.uniform(0, right_img.shape[0]),
                            self.rng.uniform(0, right_img.shape[1]),)

            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(right_img, rot_mat, right_img.shape[1::-1],
                                       flags=cv2.INTER_LINEAR)
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(right_img, trans_mat, right_img.shape[1::-1],
                                       flags=cv2.INTER_LINEAR)
        return left_img, right_img, left_disp

    def random_resize(self, left_img, right_img, left_disp):
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(left_img, None, fx=resize_scale, fy=resize_scale,
                              interpolation=cv2.INTER_LINEAR,)
        right_img = cv2.resize(right_img, None, fx=resize_scale, fy=resize_scale,
                               interpolation=cv2.INTER_LINEAR,)

        disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        disp_mask = disp_mask.astype("float32")
        disp_mask = cv2.resize(disp_mask, None, fx=resize_scale, fy=resize_scale,
                               interpolation=cv2.INTER_LINEAR,)
        left_disp = (cv2.resize(left_disp, None, fx=resize_scale, fy=resize_scale,
                                interpolation=cv2.INTER_LINEAR,) * resize_scale)
        return left_img, right_img, left_disp, disp_mask

    def random_crop(self, left_img, right_img, left_disp, disp_mask):
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(left_img, M, (self.image_width, self.image_height),
                                  flags=cv2.INTER_LINEAR, borderValue=0,)
        right_img = cv2.warpAffine(right_img, M, (self.image_width, self.image_height),
                                   flags=cv2.INTER_LINEAR, borderValue=0,)
        left_disp = cv2.warpAffine(left_disp, M, (self.image_width, self.image_height),
                                   flags=cv2.INTER_LINEAR, borderValue=0,)
        disp_mask = cv2.warpAffine(disp_mask, M, (self.image_width, self.image_height),
                                   flags=cv2.INTER_LINEAR, borderValue=0,)
        return left_img, right_img, left_disp, disp_mask

    def random_occlusion(self, right_img):
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx: cx + sx, cy - sy: cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]
        return right_img

    def random_chromatic(self, left_img, right_img):
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)
        return left_img, right_img

    def __call__(self, left_img, right_img, left_disp):
        left_img, right_img = self.random_chromatic(left_img, right_img)
        left_img, right_img, left_disp = self.random_shift(left_img, right_img, left_disp)
        left_img, right_img, left_disp, disp_mask = self.random_resize(left_img, right_img, left_disp)
        left_img, right_img, left_disp, disp_mask = self.random_crop(
            left_img, right_img, left_disp, disp_mask)
        # right_img = self.random_occlusion(right_img)
        return left_img, right_img, left_disp, disp_mask


def debug_main() -> None:
    import time
    augmentor = Augmentor(image_height=320, image_width=576, max_disp=256,
                          scale_min=0.6, scale_max=1.0, seed=int(time.time()),)

    path_list = ['./Example/000001_10_l.png',
                 './Example/000001_10_r.png',
                 './Example/000001_10_disp.png']
    save_path_list = ['./Example/000001_10_l_aug.png',
                      './Example/000001_10_r_aug.png',
                      './Example/000001_10_disp_aug.png']

    left_img = np.array(cv2.imread(path_list[0]))
    print("left image's size: ", left_img.shape)
    right_img = cv2.imread(path_list[1])
    print("right image's size: ", right_img.shape)
    disp = np.array(Image.open(path_list[2]), np.float32) / 256.0
    print("disparity image's size: ", disp.shape)
    left_img, right_img, disp, disp_mask = augmentor(left_img, right_img, disp)

    imgs_list = [left_img, right_img, (disp * 256.0).astype(np.uint16), disp_mask]

    for i in range(len(path_list)):
        cv2.imwrite(save_path_list[i], imgs_list[i])


if __name__ == "__main__":
    debug_main()
