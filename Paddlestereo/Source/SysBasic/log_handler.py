# -*- coding: utf-8 -*-
import os
import logging


class LogHandler(object):
    """docstring for LogHandler"""
    # output file setting
    LOG_FILE = 'output.log'  # log file's path
    # define some struct
    LOG_FORMAT = '[%(levelname)s] %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s'
    # LOG_FORMAT = '%(asctime)s: %(message)s'
    LOG_DATE_FORMAT = '[%a] %Y-%m-%d %H:%M:%S'

    COLOR_SEQ_HEAD = "\033[1;%dm"
    COLOR_SEQ_END = "\033[0m"

    COLOR_GREEN = 32
    COLOR_YELLOW = 33
    COLOR_RED = 31

    def __init__(self, info_format: str = None, data_format: str = None,
                 file_name: str = None) -> None:
        super().__init__()
        self.__info_format = LogHandler.LOG_FORMAT if info_format is None else info_format
        self.__data_format = LogHandler.LOG_DATE_FORMAT if data_format is None else data_format
        self.__file_name = LogHandler.LOG_FILE if file_name is None else file_name

    def init_log(self, path: str, renew: bool) -> None:
        path += self.__file_name
        if renew and os.path.exists(path):
            os.remove(path)
        logging.basicConfig(level=logging.INFO, format=self.__info_format,
                            datefmt = self.__data_format, filename = path, filemode = 'a',
                            force=True)

    def _disable_output_to_termimal(self) -> None:
        logger = logging.getLogger()
        logger.disabled = True

    def _eable_output_to_termimal(self) -> None:
        logger = logging.getLogger()
        logger.disabled = False

    @ staticmethod
    def info(data_str: str) -> None:
        print(LogHandler.COLOR_SEQ_HEAD % LogHandler.COLOR_GREEN
              + "[INFO] " + data_str + LogHandler.COLOR_SEQ_END)
        logging.info(data_str)

    @staticmethod
    def debug(data_str: str) -> None:
        print(LogHandler.COLOR_SEQ_HEAD % LogHandler.COLOR_GREEN
              + "[DEBUG] " + data_str + LogHandler.COLOR_SEQ_END)
        logging.debug(data_str)

    @ staticmethod
    def warning(data_str: str) -> None:
        print(LogHandler.COLOR_SEQ_HEAD % LogHandler.COLOR_YELLOW
              + "[WARNING] " + data_str + LogHandler.COLOR_SEQ_END)
        logging.warning(data_str)

    @ staticmethod
    def error(data_str: str) -> None:
        print(LogHandler.COLOR_SEQ_HEAD % LogHandler.COLOR_RED
              + "[ERROR] " + data_str + LogHandler.COLOR_SEQ_END)
        logging.error(data_str)
