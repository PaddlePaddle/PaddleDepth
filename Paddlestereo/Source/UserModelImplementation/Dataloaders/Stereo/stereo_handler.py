# -*- coding: utf-8 -*-
import paddle
import tifffile
import numpy as np
import os
import time
import cv2

try:
    from .stereo_augmentation import Augmentor, SparseFlowAugmentor
    from .mask_augmentation import MaskAug
    from .data_handler import DataAugmentation, ImgIO, Switch
except ImportError:
    from stereo_augmentation import Augmentor, SparseFlowAugmentor
    from mask_augmentation import MaskAug
    from data_handler import DataAugmentation, ImgIO, Switch


class StereoHandler(object):
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        self.__args = args
        self.__img_read_func, self.__label_read_func = \
            self.__read_func(args.dataset)
        self.augmentor = Augmentor(image_height=args.imgHeight,
                                   image_width=args.imgWidth,
                                   max_disp=args.dispNum,
                                   scale_min=0.6, scale_max=1.0,
                                   seed=int(time.time()),)
        self.mask_aug = MaskAug(args.imgHeight, args.imgWidth, ratio=0.15)

    def _get_img_read_func(self):
        return self.__img_read_func, self.__label_read_func

    def _read_data(self, left_img_path: str, right_img_path: str, gt_dsp_path: str) -> tuple:
        left_img = np.array(self.__img_read_func(left_img_path, True))
        right_img = np.array(self.__img_read_func(right_img_path, True))
        gt_dsp = np.array(self.__label_read_func(gt_dsp_path))
        return left_img, right_img, gt_dsp

    def _read_training_data(self, left_img_path: str,
                            right_img_path: str,
                            gt_dsp_path: str) -> tuple:
        args = self.__args
        left_img, right_img, gt_dsp = self._read_data(left_img_path, right_img_path, gt_dsp_path)

        if self.__args.modelName == "RAFT_STEREO":
            self.augmentor = SparseFlowAugmentor(crop_size=[args.imgHeight, args.imgWidth])
            left_img = left_img.astype(np.uint8)
            right_img = right_img.astype(np.uint8)
            disp = gt_dsp.astype(np.float32)
            valid = disp > 0.0
            flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

            if len(left_img.shape) == 2:
                left_img = np.tile(left_img[...,None], (1, 1, 3))
                right_img = np.tile(right_img[...,None], (1, 1, 3))
            else:
                left_img = left_img[..., :3]
                right_img = right_img[..., :3]

            left_img, right_img, flow, valid = self.augmentor(left_img, right_img, flow, valid)
            left_img = left_img.transpose(2, 0, 1).astype('float32')
            right_img = right_img.transpose(2, 0, 1).astype('float32')
            flow = flow.transpose(2, 0, 1).astype('float32')
            flow = flow[:1]
            flow[np.isinf(flow)] = 0
            gt_dsp = np.stack([np.squeeze(flow, axis=0), valid.astype('float32')], axis=-1)
        else:
            gt_dsp = np.expand_dims(gt_dsp, axis=2)
            left_img, right_img, gt_dsp = DataAugmentation.random_crop(
                [left_img, right_img, gt_dsp],
                left_img.shape[1], left_img.shape[0], args.imgWidth, args.imgHeight)
            
            left_img, right_img = DataAugmentation.standardize(left_img), \
                DataAugmentation.standardize(right_img)


            left_img, right_img = left_img.transpose(2, 0, 1), right_img.transpose(2, 0, 1)
            gt_dsp = np.squeeze(gt_dsp, axis=2)
            #left_img_mask = left_img_mask.transpose(2, 0, 1)

            gt_dsp[np.isinf(gt_dsp)] = 0
        return left_img, right_img, gt_dsp, left_img
        # return left_img_mask, right_img, \
        #    gt_dsp.astype('float32') * mask.astype('float32'), left_img.astype('float32')
        #    (gt_dsp * mask).astype('float32'), left_img.astype('float32')
        # return left_img, right_img, gt_dsp

    def _img_padding(self, left_img: np.array, right_img: np.array) -> tuple:
        # pading size
        args = self.__args

        if left_img.shape[0] < args.imgHeight:
            padding_height = args.imgHeight
            padding_width = args.imgWidth
        else:
            padding_height = self._padding_size(left_img.shape[0])
            padding_width = self._padding_size(left_img.shape[1])

        top_pad = padding_height - left_img.shape[0]
        left_pad = padding_width - right_img.shape[1]

        # pading
        left_img = np.lib.pad(left_img, ((top_pad, 0), (0, left_pad), (0, 0)),
                              mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((top_pad, 0), (0, left_pad), (0, 0)),
                               mode='constant', constant_values=0)

        return left_img, right_img, top_pad, left_pad

    def _read_testing_data(self, left_img_path: str,
                           right_img_path: str,
                           gt_dsp_path: str) -> object:
        args = self.__args

        left_img = np.array(self.__img_read_func(left_img_path, True))
        right_img = np.array(self.__img_read_func(right_img_path, True))

        if self.__args.modelName == "RAFT_STEREO":
            left_img = left_img.astype(np.float32)
            right_img = right_img.astype(np.float32)
            top_pad, left_pad = 0, 0
        else:           
            left_img = DataAugmentation.standardize(left_img)
            right_img = DataAugmentation.standardize(right_img)

            left_img, right_img, top_pad, left_pad = self._img_padding(left_img, right_img)

        left_img = left_img.transpose(2, 0, 1)
        right_img = right_img.transpose(2, 0, 1)

        gt_dsp = None
        if gt_dsp_path is not None:
            gt_dsp = np.array(self.__label_read_func(gt_dsp_path))
            gt_dsp = np.lib.pad(gt_dsp, ((top_pad, 0), (0, left_pad)),
                                mode='constant', constant_values=0)

        name = self._get_name(args.dataset, left_img_path)

        return left_img, right_img, gt_dsp, top_pad, left_pad, name

    def __read_func(self, dataset_name: str) -> object:
        img_read_func = None
        label_read_func = None
        for case in Switch(dataset_name):
            if case('US3D'):
                img_read_func, label_read_func = tifffile.imread, tifffile.imread
                break
            if case('kitti2012') or case('kitti2015'):
                img_read_func, label_read_func = ImgIO.read_img, self._read_png_disp
                break
            if case('eth3d') or case('middlebury') or case('sceneflow'):
                img_read_func, label_read_func = ImgIO.read_img, self._read_pfm_disp
                break
            if case('crestereo'):
                img_read_func, label_read_func = ImgIO.read_img, self._read_cre_disp
                break
            if case('rob'):
                img_read_func, label_read_func = ImgIO.read_img, self._read_rob_disp
                break
            if case():
                print("The model's name is error!!!")

        return img_read_func, label_read_func

    def get_data(self, left_img_path: str, right_img_path: str,
                 gt_dsp_path: str, is_training: bool) -> tuple:
        if is_training:
            return self._read_training_data(left_img_path, right_img_path, gt_dsp_path)
        return self._read_testing_data(left_img_path, right_img_path, gt_dsp_path)

    @staticmethod
    def expand_batch_size_dims(left_img: np.array, right_img: np.array, gt_dsp: np.array,
                               top_pad: int, left_pad: int, name: str) -> tuple:
        left_img = np.expand_dims(left_img, axis=0)
        right_img = np.expand_dims(right_img, axis=0)
        gt_dsp, top_pad, left_pad, name = [gt_dsp], [top_pad], [left_pad], [name]

        left_img = paddle.to_tensor(left_img)
        right_img = paddle.to_tensor(right_img)
        return left_img, right_img, gt_dsp, top_pad, left_pad, name

    @staticmethod
    def _read_png_disp(path: str) -> np.array:
        gt_dsp = ImgIO.read_img(path)
        gt_dsp = np.squeeze(gt_dsp, 2)
        gt_dsp = np.ascontiguousarray(
            gt_dsp, dtype=np.float32) / float(StereoHandler._DEPTH_DIVIDING)
        return gt_dsp

    @staticmethod
    def _read_cre_disp(path: str) -> np.array:
        gt_dsp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return gt_dsp.astype(np.float32) / 32

    @staticmethod
    def _read_pfm_disp(path: str) -> np.array:
        gt_dsp, _ = ImgIO.read_pfm(path)
        return gt_dsp

    @staticmethod
    def _read_rob_disp(path: str) -> np.array:
        file_type = os.path.splitext(path)[-1]
        if file_type == ".png":
            gt_dsp = StereoHandler._read_png_disp(path)
        else:
            gt_dsp = StereoHandler._read_pfm_disp(path)
        return gt_dsp

    @staticmethod
    def _padding_size(value: int, base: int = 64) -> int:
        off_set = 1
        return value // base + off_set

    @staticmethod
    def _get_name(dataset_name: str, path: str) -> str:
        name = ""
        if dataset_name in {"eth3d", "middlebury"}:
            off_set = 1
            pos = path.rfind('/')
            name = path[:pos]
            pos = name.rfind('/')
            name = name[pos + off_set:]
        return name
