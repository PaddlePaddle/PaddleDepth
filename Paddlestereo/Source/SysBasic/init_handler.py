# -*- coding: utf-8 -*-
import os

from SysBasic.log_handler import LogHandler as log
from FileHandler.file_handler import FileHandler

from .device_manager import DeviceManager


class InitProgram(object):
    """docstring for InitProgram"""

    def __init__(self, args) -> None:
        super().__init__()
        self.__args = args

    def __build_result_directory(self) -> None:
        args = self.__args
        FileHandler.mkdir(args.outputDir)
        FileHandler.mkdir(args.modelDir)
        FileHandler.mkdir(args.resultImgDir)
        FileHandler.mkdir(args.log)

    def __show_args(self) -> None:
        args = self.__args
        log.info("The hyper-parameters are set as follows:")
        log.info('├── mode: ' + str(args.mode))
        log.info('├── dataset: ' + str(args.dataset))
        log.info('├── trainListPath: ' + str(args.trainListPath))
        log.info('├── valListPath: ' + str(args.valListPath))
        log.info('├── outputDir: ' + str(args.outputDir))
        log.info('├── modelDir: ' + str(args.modelDir))
        log.info('├── resultImgDir: ' + str(args.resultImgDir))
        log.info('├── log: ' + str(args.log))
        log.info('├── gpu: ' + str(args.gpu))
        log.info('├── dist: ' + str(args.dist))
        log.info('├── dataloaderNum: ' + str(args.dataloaderNum))
        log.info('├── auto_save_num: ' + str(args.auto_save_num))
        log.info('├── sampleNum: ' + str(args.sampleNum))
        log.info('├── maxEpochs: ' + str(args.maxEpochs))
        log.info('├── batchSize: ' + str(args.batchSize))
        log.info('├── lr: ' + str(args.lr))
        log.info('├── pretrain: ' + str(args.pretrain))
        log.info('├── modelName: ' + str(args.modelName))
        log.info('├── imgNum: ' + str(args.imgNum))
        log.info('├── valImgNum: ' + str(args.valImgNum))
        log.info('├── imgWidth: ' + str(args.imgWidth))
        log.info('└── imgHeight: ' + str(args.imgHeight))

    def __check_args(self) -> bool:
        res = True
        log.info('Begin to check the args')
        if not os.path.exists(self.__args.trainListPath):
            log.error('the training list is not existed!')
            res = False
        if not os.path.exists(self.__args.valListPath):
            log.warning('the val list is not existed!')
        if os.path.isfile(self.__args.outputDir):
            log.error("A file was passed as `--outputDir`, please pass a directory!")
            res = False
        if os.path.isfile(self.__args.modelDir):
            log.warning("A file was passed as `--modelDir`, suggest to pass a directory!")
        if os.path.isfile(self.__args.resultImgDir):
            log.error("A file was passed as `--resultImgDir`, please pass a directory!")
            res = False
        if os.path.isfile(self.__args.log):
            log.error("A file was passed as `--log`, please pass a directory!")
            res = False
        if res:
            log.info('Finish checking the args')
        else:
            log.info('Error in the process of checking args')
        return res

    def __check_env(self) -> bool:
        log.info('Begin to check the env')
        return DeviceManager.check_cuda(self.__args)

    def init_program(self) -> bool:
        args = self.__args
        self.__build_result_directory()
        log().init_log(args.outputDir, args.pretrain)
        log.info("Welcome to use the JackFramework")
        self.__show_args()
        res = self.__check_args() and self.__check_env()
        if not res:
            log.error("Failed in the init programs")
        return res
