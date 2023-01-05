# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
import paddle

from Algorithm import ListHandler


class ResultContainer(object):
    def __init__(self):
        super().__init__()
        self._tower_loss, self._tower_acc = None, None
        self._ave_tower_loss, self._ave_tower_acc = None, None
        self._tower_loss_iteration, self._tower_acc_iteration = [], []

    @property
    def tower_loss(self) -> list:
        return self._tower_loss

    @property
    def tower_acc(self) -> list:
        return self._tower_acc

    @property
    def ave_tower_loss(self) -> list:
        return self._ave_tower_loss

    @property
    def ave_tower_acc(self) -> list:
        return self._ave_tower_acc

    @property
    def tower_loss_iteration(self) -> list:
        return self._tower_loss_iteration

    @property
    def tower_acc_iteration(self) -> list:
        return self._tower_acc_iteration

    def init_result(self):
        self._tower_loss, self._tower_acc = None, None
        self._ave_tower_loss, self._ave_tower_acc = None, None

    def cal_loss(self, total_iteration: int) -> None:
        self._tower_loss = ListHandler.double_list_add(self._tower_loss_iteration, self._tower_loss)
        self._ave_tower_loss = ListHandler.double_list_div(self._tower_loss, total_iteration)

    def cal_acc(self, total_iteration) -> None:
        self._tower_acc = ListHandler.double_list_add(self._tower_acc_iteration, self._tower_acc)
        self._ave_tower_acc = ListHandler.double_list_div(self._tower_acc, total_iteration)

    def cal_tower_loss_acc(self, total_iteration: int) -> None:
        self.cal_loss(total_iteration)
        self.cal_acc(total_iteration)

    def init_tower_loss_and_tower_acc(self):
        self._tower_loss_iteration, self._tower_acc_iteration = [], []

    def append_iteration_loss(self, data: paddle.tensor) -> None:
        self._tower_loss_iteration.append(data)

    def append_iteration_acc(self, data: paddle.tensor) -> None:
        self._tower_acc_iteration.append(data)

    def append_iteration_loss_acc(self, loss_data: paddle.tensor, acc_data: paddle.tensor) -> None:
        self.append_iteration_loss(loss_data)
        self.append_iteration_acc(acc_data)
