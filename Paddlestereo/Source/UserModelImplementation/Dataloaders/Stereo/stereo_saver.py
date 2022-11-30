# -*- coding: utf-8 -*-
import numpy as np
import cv2

try:
    from .data_handler import ImgIO, Switch, FileHandler
except ImportError:
    from data_handler import ImgIO, Switch, FileHandler


class StereoSaver(object):
    """docstring for StereoSaver"""
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        super().__init__()
        self.__args = args

    def _save_per_output(self, idx: int, batch_size: int, tmp_res: np.array,
                         img_id: int, dataset_name: str, names: list, ttimes: float) -> None:
        for case in Switch(dataset_name):
            if case('US3D'):
                print("Unsupport the US3D dataset!!!")
                break
            if case('kitti2012') or case('kitti2015') or case('sceneflow'):
                name = batch_size * img_id + idx
                self._save_kitti_test_data(tmp_res, name)
                break
            if case('eth3d'):
                name = names[idx]
                self._save_eth3d_test_data(tmp_res, name, ttimes)
                break
            if case('middlebury'):
                name = names[idx]
                self._save_middlebury_test_data(tmp_res, name, ttimes)
                break
            if case():
                print("The model's name is error!!!")

    def _save_kitti_test_data(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num)
        img = self._depth2img(img)
        self._save_png_img(path, img)

    def _save_eth3d_test_data(self, img: np.array,
                              name: str, ttimes: str) -> None:
        args = self.__args
        path = args.resultImgDir + name + '.pfm'
        ImgIO.write_pfm(path, img)
        path = args.resultImgDir + name + '.txt'
        with open(path, 'w') as f:
            f.write("runtime " + str(ttimes))
            f.close()

    def _save_middlebury_test_data(self, img: np.array,
                                   name: str, ttimes: str) -> None:
        args = self.__args
        folder_name = args.resultImgDir + name + '/'
        FileHandler.mkdir(folder_name)
        method_name = "disp0" + args.modelName + "_RVC.pfm"
        path = folder_name + method_name
        ImgIO.write_pfm(path, img)

        time_name = "time" + args.modelName + "_RVC.txt"
        path = folder_name + time_name
        with open(path, 'w') as f:
            f.write(str(ttimes))
            f.close()

    def save_output_by_path(self, disp: np.array, supplement: list, path: str) -> None:
        batch_size, _, _ = disp.shape
        assert batch_size == 1
        top_pads = supplement[0]
        left_pads = supplement[1]
        for i in range(batch_size):
            tmp_res = disp[i, :, :]
            top_pad = top_pads[i]
            left_pad = left_pads[i]
            tmp_res = self._crop_test_img(tmp_res, top_pad, left_pad)
            tmp_res = self._depth2img(tmp_res)
            self._save_png_img(path, tmp_res)

    def save_output(self, disp: np.array, img_id: int, dataset_name: str,
                    supplement: list, ttimes: float) -> None:
        batch_size, _, _ = disp.shape
        top_pads = supplement[0]
        left_pads = supplement[1]
        names = supplement[2]

        for i in range(batch_size):
            tmp_res = disp[i, :, :]
            top_pad = top_pads[i]
            left_pad = left_pads[i]
            tmp_res = self._crop_test_img(tmp_res, top_pad, left_pad)
            self._save_per_output(i, batch_size, tmp_res,
                                  img_id, dataset_name, names, ttimes)

    @staticmethod
    def _crop_test_img(img: np.array, top_pad: int, left_pad: int) -> np.array:
        if top_pad > 0 and left_pad > 0:
            img = img[top_pad:, : -left_pad]
        elif top_pad > 0:
            img = img[top_pad:, :]
        elif left_pad > 0:
            img = img[:, :-left_pad]
        return img

    @staticmethod
    def _generate_output_img_path(dir_path: str, num: str,
                                  filename_format: str = "%06d_10",
                                  img_type: str = ".png"):
        return dir_path + filename_format % num + img_type

    @staticmethod
    def _depth2img(img: np.array) -> np.array:
        img = np.array(img)
        img = (img * float(StereoSaver._DEPTH_DIVIDING)).astype(np.uint16)
        return img

    @staticmethod
    def _save_png_img(path: str, img: np.array) -> None:
        # save the png file
        cv2.imwrite(path, img)
