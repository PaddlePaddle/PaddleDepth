# -*- coding: UTF-8 -*-
from collections.abc import Callable

from SysBasic import Switch
from SysBasic import LogHandler as log

from .test_proc import TestProc
from .train_proc import TrainProc
# from .background import BackGround


def mode_selection(args: object, user_inference_func: object, mode: str) -> Callable:
    mode_func = None
    for case in Switch(mode):
        if case('train'):
            log.info("Enter training mode")
            mode_func = TrainProc(args, user_inference_func, True).exec
            break
        if case('test'):
            log.info("Enter testing mode")
            mode_func = TestProc(args, user_inference_func, False).exec
            break
        if case('background'):
            log.info("Enter background mode")
            # mode_func = BackGround(args, user_inference_func, False).exec
            break
        if case('online'):
            log.info("Enter online mode")
            break
        if case('reinforcement_learning'):
            log.info("Enter reinforcement learning mode")
            break
        if case(''):
            log.error("The mode's name is error!!!")
    return mode_func
