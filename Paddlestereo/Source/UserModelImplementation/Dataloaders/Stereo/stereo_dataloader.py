# -*- coding: utf-8 -*-
import time

from SysBasic import ResultStr
from Template import DataHandlerTemplate
from SysBasic import LogHandler as log

from .stereo_dataset import StereoDataset
from .stereo_saver import StereoSaver
from .stereo_handler import StereoHandler


class StereoDataloader(DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    MODEL_ID = 0                                       # Model
    ID_IMG_L, ID_IMG_R, ID_DISP, ID_MASK = 0, 1, 2, 3  # traning
    ID_TOP_PAD, ID_LEFT_PAD, ID_NAME = 3, 4, 5         # Test

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__saver = StereoSaver(args)
        self.__data_handler = StereoHandler(args)
        self.__start_time = 0

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        self.__train_dataset = StereoDataset(args, args.trainListPath, is_training)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        args = self.__args
        self.__val_dataset = StereoDataset(args, args.valListPath, False)
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [batch_data[self.ID_IMG_L], batch_data[self.ID_IMG_R]],\
                [batch_data[self.ID_DISP], batch_data[self.ID_MASK]]
            #    [batch_data[self.ID_DISP]]
            # return input_data, supplement
        return [batch_data[self.ID_IMG_L], batch_data[self.ID_IMG_R]], \
            [batch_data[self.ID_TOP_PAD], batch_data[self.ID_LEFT_PAD], batch_data[self.ID_NAME]]

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        info_str = self.__result_str.training_result_str(
            epoch, loss[self.MODEL_ID], acc[self.MODEL_ID], duration, True)
        log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        info_str = self.__result_str.training_result_str(
            epoch, loss[self.MODEL_ID], acc[self.MODEL_ID], duration, False)
        log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args = self.__args
        off_set = 1
        last_position = len(output_data) - off_set
        # last_position = 0

        if model_id == self.MODEL_ID:
            self.__saver.save_output(output_data[last_position].cpu().detach().numpy(),
                                     img_id, args.dataset, supplement,
                                     time.time() - self.__start_time)

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        return self.__result_str.training_intermediate_result(
            epoch, loss[self.MODEL_ID], acc[self.MODEL_ID])

    # optional for background
    def load_test_data(self, cmd: str) -> tuple:
        cmd_list = cmd.split(',')
        path_num = 3
        if len(cmd_list) != path_num:
            return None

        left_img_path, right_img_path, _ = cmd_list
        left_img, right_img, gt_dsp, top_pad, left_pad, name = \
            self.__data_handler.get_data(
                left_img_path, right_img_path, None, False)
        left_img, right_img, gt_dsp, top_pad, left_pad, name = \
            self.__data_handler.expand_batch_size_dims(
                left_img, right_img, gt_dsp, top_pad, left_pad, name)
        return left_img, right_img, gt_dsp, top_pad, left_pad, name

    def save_test_data(self, output_data: list, supplement: list, cmd: str, model_id: int) -> None:
        cmd_list = cmd.split(',')
        path_num = 3
        if len(cmd_list) != path_num:
            return None

        _, _, save_path = cmd_list
        save_path = save_path.rstrip("\n")

        off_set = 1
        last_position = len(output_data) - off_set
        # last_position = 0
        if model_id == self.MODEL_ID:
            self.__saver.save_output_by_path(output_data[last_position].cpu().detach().numpy(),
                                             supplement, save_path)
