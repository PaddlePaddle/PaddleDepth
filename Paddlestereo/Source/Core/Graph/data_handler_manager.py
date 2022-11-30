# -*- coding: utf-8 -*-
import paddle
from SysBasic import LogHandler as log
from ._user_dataloader import UserDataloader


class DataHandlerManager(UserDataloader):
    """docstring for ClassName"""

    def __init__(self, args: object, jf_data_handler: object) -> None:
        super().__init__(args, jf_data_handler)
        self.__args = args
        self.__training_dataloader = self.__check_training_dataloader()
        self.__val_dataloader = self.__check_val_dataloader()

    @property
    def training_dataloader(self) -> object:
        return self.__training_dataloader

    @property
    def val_dataloader(self) -> object:
        return self.__val_dataloader

    def get_dataloader(self, is_training: bool) -> object:
        return self.training_dataloader if is_training else self.val_dataloader

    def __check_training_dataloader(self) -> object:
        log.info("Begin loading the training dataset")
        if self.__args.imgNum > 0:
            is_training = self.__args.mode != 'test'
            training_dataloader = self.__init_training_dataloader(is_training)
        else:
            log.warning("The training images is 0")
            training_dataloader = None
        log.info("Finish constructing the training dataloader")
        return training_dataloader

    def __init_training_dataloader(self, is_training: bool) -> tuple:
        training_dataset = self.user_get_train_dataset(is_training)

        if self.__args.mode == 'test':
            shuffle = False
        else:
            shuffle = True

        batch_sampler = paddle.io.DistributedBatchSampler(
            training_dataset, batch_size=self.__args.batchSize, shuffle=shuffle, drop_last=False)
        training_dataloader = paddle.io.DataLoader(training_dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=self.__args.dataloaderNum)
        return training_dataloader

    def __init_val_dataloader(self) -> object:
        val_dataset = self.user_get_val_dataset()
        val_dataloader = paddle.io.DataLoader(val_dataset,
                                              batch_size=self.__args.batchSize,
                                              shuffle=False,
                                              num_workers=self.__args.dataloaderNum,
                                              drop_last=True)
        return val_dataloader

    def __check_val_dataloader(self) -> object:
        log.info("Begin loading the val dataset")
        if self.__args.valImgNum > 0:
            val_dataloader = self.__init_val_dataloader()
        else:
            log.warning("The val images is 0")
            val_dataloader = None
        log.info("Finish constructing the val dataloader")
        return val_dataloader
