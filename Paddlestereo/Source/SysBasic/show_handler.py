# -*- coding: utf-8 -*-
import time
from functools import wraps
from .processbar import ShowProcess
from .device_manager import DeviceManager


class ShowHandler(object):
    __ShowHandler = None
    __TENSORBOARD_HANDLER = None
    __PROCESS_BAR, __START_TIME = None, None
    __DURATION, __REST_TIME = None, None
    __DEFAULT_RANK_ID = 0
    __RANK = -1

    def __init__(self) -> None:
        super().__init__()
        DeviceManager.init_parallel_env()
        ShowHandler.__RANK = DeviceManager.get_local_rank()

    @staticmethod
    def init_show_setting(training_iteration: int, bar_info: str) -> None:
        ShowHandler.__PROCESS_BAR = ShowProcess(training_iteration, bar_info)
        ShowHandler.__START_TIME = time.time()

    @staticmethod
    def calculate_ave_runtime(total_iteration: int, training_iteration: int) -> None:
        ShowHandler.__DURATION = (time.time() - ShowHandler.__START_TIME) / total_iteration
        ShowHandler.__REST_TIME = (training_iteration - total_iteration) * ShowHandler.__DURATION

    @staticmethod
    def stop_show_setting() -> None:
        ShowHandler.__PROCESS_BAR.close()

    @staticmethod
    def duration():
        return time.time() - ShowHandler.__START_TIME

    @staticmethod
    def update_show_bar(info_str: str) -> None:
        ShowHandler.__PROCESS_BAR.show_process(show_info=info_str,
                                               rest_time=ShowHandler.__REST_TIME,
                                               duration=ShowHandler.__DURATION)

    @classmethod
    def show_method(cls, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if cls.__RANK == cls.__DEFAULT_RANK_ID or cls.__RANK is None:
                func(*args, **kwargs)

        return wrapped_func
