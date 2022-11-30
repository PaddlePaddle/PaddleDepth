# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class DataHandlerTemplate(object):
    """docstring for DataHandlerTemplate"""
    __metaclass__ = ABCMeta

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__args = args

    @abstractmethod
    def get_train_dataset(self, path: str, is_training: bool) -> object:
        # return dataset
        pass

    @abstractmethod
    def get_val_dataset(self, path: str) -> object:
        # return val dataset
        pass

    @abstractmethod
    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        # return input_data, label_data
        pass

    @abstractmethod
    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        pass

    @abstractmethod
    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        pass

    @abstractmethod
    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        pass

    @abstractmethod
    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        # return data+str
        pass

    # optional
    def load_test_data(self, cmd: str) -> tuple:
        pass

    def save_test_data(self, output_data: list, supplement: list, cmd: str, model_id: int) -> None:
        pass
