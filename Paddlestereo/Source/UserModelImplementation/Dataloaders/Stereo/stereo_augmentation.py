# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

from paddle.vision.transforms import ColorJitter, functional, Compose

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


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def adjust_gamma(
    img: Image.Image,
    gamma: float,
    gain: float = 1.0,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    input_mode = img.mode
    img = img.convert("RGB")
    gamma_map = [int((255 + 1 - 1e-3) * gain * pow(ele / 255.0, gamma)) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.4, do_flip=False, yjitter=False, saturation_range=[0,1.4], gamma=[1,1,1,1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, valid


    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid


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
