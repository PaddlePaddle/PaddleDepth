# -*- coding: utf-8 -*-
import os
from abc import ABCMeta, abstractmethod

import paddle
from paddle.distributed import fleet
import SysBasic.define as sys_define
from SysBasic import LogHandler as log
from FileHandler import ModelSaver
from Template import ModelHandlerTemplate

from ._user_model import UserModel


class MetaOps(UserModel):
    __metaclass__ = ABCMeta
    __OPT_LR_GROUP_ID = 0

    def __init__(self, args: object, jf_model: ModelHandlerTemplate) -> None:
        super().__init__(args, jf_model)
        self.__args = args
        self.__init_training_graph()

    def __init_training_graph(self) -> None:
        self.user_init_model()
        self.user_init_optimizer()
        self._pass_model2device()
        self.count_parameter_num()

    def _init_ddp_model(self) -> None:
        fleet.init(is_collective=True)
        assert self._model is not None and self._opt is not None
        for i, model_item in enumerate(self._model):
            self._model[i] = fleet.distributed_model(model_item)
        for i, opt_item in enumerate(self._opt):
            self._opt[i] = fleet.distributed_optimizer(opt_item)

    def _pass_model2device(self) -> None:
        log.info("Loading model to GPUs!")
        self._init_ddp_model()
        log.info("Successfully loaded the model into GPUs!")

    def _variable2tensor(self, data: list) -> list:
        res = []
        for data_item in data:
            res.append(data_item.numpy())
        return res

    def _restore_model_opt(self, checkpoint: dict) -> None:
        for i, _ in enumerate(self._model):
            if not self.user_load_model(checkpoint, i):
                ModelSaver.load_model(self._model[i], checkpoint, i)
            if not self.user_load_opt(checkpoint, i):
                ModelSaver.load_opt(self._opt[i], checkpoint, i)

    def show_lr_scheduler_info(self, idx: int) -> None:
        log.info((f'Model_{idx} Current lr: ' +
                  str(self._opt[idx].get_lr())))

    def count_parameter_num(self) -> None:
        for i, model_item in enumerate(self._model):
            num_params = sum(paddle.numel(param) for param in model_item.parameters())
            log.info(f'Model {str(i)}' + f': The total parameter - {num_params}')

    def adjust_lr_scheduler(self, loss: list, epoch: int) -> None:
        for i, sch_item in enumerate(self._sch):
            if sch_item is not None:
                self.user_lr_scheduler(sch_item, loss, i, epoch)
                self.show_lr_scheduler_info(i)

    def restore_model(self) -> None:
        checkpoint_path = ModelSaver.get_check_point_path(self.__args.modelDir)
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = ModelSaver.load_checkpoint(checkpoint_path)
            self._restore_model_opt(checkpoint)
        else:
            log.warning(f"no checkpoint found at '{checkpoint_path}'")

    def save_model(self, epoch: int) -> None:
        assert len(self._model) == len(self._opt)
        file_name = sys_define.CHECK_POINT_NAME % epoch
        model_dict = self.user_save_model(epoch)
        if model_dict is None:
            model_dict = ModelSaver.construct_model_dict(epoch, self._model, self._opt)
        ModelSaver.save(self.__args.modelDir, file_name, model_dict)

    def set_model_mode(self, is_training: bool = True) -> None:
        assert self._model is not None
        for i, _ in enumerate(self._model):
            if is_training:
                self._model[i].train()
            else:
                self._model[i].eval()

    def stop_gradient(self, label_data: list) -> None:
        for data_item in label_data:
            data_item.stop_gradient = True

    @abstractmethod
    def exec(self, input_data: list, label_data: list, is_training: bool = True) -> list:
        pass
