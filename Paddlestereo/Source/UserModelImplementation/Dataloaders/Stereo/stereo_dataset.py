# -*- coding: utf-8 -*-
from paddle.io import Dataset
import pandas as pd

try:
    from .stereo_handler import StereoHandler
except ImportError:
    from stereo_handler import StereoHandler


class StereoDataset(Dataset):
    """docstring for DFCStereoDataset"""
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object, list_path: str,
                 is_training: bool = False) -> None:
        self.__args = args
        self.__is_training = is_training
        self.__data_handler = StereoHandler(args)

        input_dataframe = pd.read_csv(list_path)
        self.__left_img_path = input_dataframe["left_img"].values
        self.__right_img_path = input_dataframe["right_img"].values
        self.__gt_dsp_path = input_dataframe["gt_disp"].values

        if is_training:
            self.__get_path = self._get_training_path
            self.__data_steam = list(zip(
                self.__left_img_path, self.__right_img_path, self.__gt_dsp_path))
        else:
            self.__get_path = self._get_testing_path
            self.__data_steam = list(zip(
                self.__left_img_path, self.__right_img_path))

    def __getitem__(self, idx: int):
        left_img_path, right_img_path, gt_dsp_path = self.__get_path(idx)
        return self.__data_handler.get_data(
            left_img_path, right_img_path, gt_dsp_path, self.__is_training)

    def _get_training_path(self, idx: int) -> list:
        return self.__left_img_path[idx],\
            self.__right_img_path[idx], self.__gt_dsp_path[idx]

    def _get_testing_path(self, idx: int) -> list:
        return self.__left_img_path[idx], self.__right_img_path[idx], \
            self.__gt_dsp_path[idx]

    def __len__(self):
        return len(self.__data_steam)
