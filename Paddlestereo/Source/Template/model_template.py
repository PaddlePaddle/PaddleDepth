# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ModelHandlerTemplate(object):
    __metaclass__ = ABCMeta

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__args = args

    @abstractmethod
    def get_model(self) -> list:
        # return model's list
        pass

    @abstractmethod
    def inference(self, model: object, input_data: list, model_id: int) -> list:
        pass

    @abstractmethod
    def optimizer(self, model: list, lr: float) -> list:
        # return list of optimizer
        # return list of scheduler
        pass

    @abstractmethod
    def lr_scheduler(self, sch: object, ave_loss: float, sch_id: int, epoch: str) -> None:
        pass

    @abstractmethod
    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return accuracy's list
        pass

    @abstractmethod
    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss's list
        pass

    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    def post_process(self, epoch: int, rank: object, ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
