# -*- coding: utf-8 -*-
import os
import paddle
from SysBasic.log_handler import LogHandler as log
import paddle.distributed as dist


class DeviceManager(object):
    __DEVICE_MANAGER = None
    _NONE_CUDA_DEVICE = 0

    def __init__(self, args: object):
        super().__init__()
        self.__args = args

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__DEVICE_MANAGER is None:
            cls.__DEVICE_MANAGER = object.__new__(cls)
        return cls.__DEVICE_MANAGER

    @staticmethod
    def check_cuda(args):
        if args.gpu == 0:
            log.info('We will use cpu!')
            return True

        if paddle.device.cuda.device_count() == DeviceManager._NONE_CUDA_DEVICE:
            log.error("Torch is reporting that CUDA isn't available")
            return False

        log.info(f"We detect the gpu device: {paddle.device.cuda.get_device_name(0)}")
        log.info(f"We detect the number of gpu device: {paddle.device.cuda.device_count()}")
        args, res_bool = DeviceManager.check_cuda_count(args)
        return res_bool

    @staticmethod
    def check_cuda_count(args) -> object:
        res_bool = True
        if paddle.device.cuda.device_count() < args.gpu:
            log.warning("The setting of GPUs is more than actually owned GPUs: " +
                        f"{args.gpu} vs {paddle.device.cuda.device_count()}")
            log.info("We will use all actually owned GPUs.")
            args.gpu = paddle.device.cuda.device_count()

        return args, res_bool

    @staticmethod
    def init_parallel_env() -> None:
        dist.init_parallel_env()

    @staticmethod
    def get_local_rank() -> int:
        return dist.ParallelEnv().local_rank
