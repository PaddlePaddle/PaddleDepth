# -*- coding: utf-8 -*-
import os
import paddle

import SysBasic.define as sys_def
from SysBasic import LogHandler as log
from .file_handler import FileHandler


class ModelSaver(object):
    """docstring for ModelSaver"""
    ROW_ONE = 1
    FIST_SAVE_TIME = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def __get_model_name(file_path: str) -> str:
        str_line = FileHandler.get_line(file_path, ModelSaver.ROW_ONE)
        return str_line[len(sys_def.LAST_MODEL_NAME):len(str_line)]

    @staticmethod
    def __check_list(file_dir: str, fd_checkpoint_list: object) -> object:
        str_line = FileHandler.get_line_fd(fd_checkpoint_list, 0)
        if str_line[: len(sys_def.LAST_MODEL_NAME)] != sys_def.LAST_MODEL_NAME:
            log.warning("The checklist file is wrong! We will rewrite this file")
            FileHandler.close_file(fd_checkpoint_list)
            os.remove(file_dir + sys_def.CHECK_POINT_LIST_NAME)
            fd_checkpoint_list = None
        return fd_checkpoint_list

    @staticmethod
    def __write_check_point_list_title(file_dir: str) -> object:
        if ModelSaver.FIST_SAVE_TIME:
            FileHandler.remove_file(file_dir + sys_def.CHECK_POINT_LIST_NAME)
            ModelSaver.FIST_SAVE_TIME = False
            return None

        fd_checkpoint_list = FileHandler.open_file(file_dir + sys_def.CHECK_POINT_LIST_NAME)
        fd_checkpoint_list = ModelSaver.__check_list(file_dir, fd_checkpoint_list)
        return fd_checkpoint_list

    @staticmethod
    def __write_new_check_point_file(file_dir: str, file_name: str) -> None:
        fd_checkpoint_list = FileHandler.open_file(file_dir + sys_def.CHECK_POINT_LIST_NAME)
        FileHandler.write_file(fd_checkpoint_list, sys_def.LAST_MODEL_NAME + file_name)
        FileHandler.write_file(fd_checkpoint_list, file_name)
        FileHandler.close_file(fd_checkpoint_list)

    @staticmethod
    def __write_old_check_point_file(fd_checkpoint_list: object,
                                     file_dir: str, file_name: str) -> None:
        fd_checkpoint_temp_list = FileHandler.open_file(
            file_dir + sys_def.CHECK_POINT_LIST_NAME + '.temp')

        FileHandler.write_file(fd_checkpoint_temp_list, sys_def.LAST_MODEL_NAME + file_name)
        FileHandler.copy_file(fd_checkpoint_list, fd_checkpoint_temp_list, 1)
        FileHandler.write_file(fd_checkpoint_temp_list, file_name)

        FileHandler.close_file(fd_checkpoint_list)
        FileHandler.close_file(fd_checkpoint_temp_list)

        os.remove(file_dir + sys_def.CHECK_POINT_LIST_NAME)
        os.rename(file_dir + sys_def.CHECK_POINT_LIST_NAME + '.temp',
                  file_dir + sys_def.CHECK_POINT_LIST_NAME)

    @staticmethod
    def __load_model_folder(path: str) -> str:
        log.info("Begin loading checkpoint from this folder: '{}'".format(path))
        checkpoint_list_file = path + sys_def.CHECK_POINT_LIST_NAME
        if not os.path.isfile(checkpoint_list_file):
            log.warning("We don't find the checkpoint list file: '{}'!".format(checkpoint_list_file))
            checkpoint_path = None
        else:
            checkpoint_path = path + ModelSaver.__get_model_name(checkpoint_list_file)
            log.info("Get the path of model: '{}'".format(checkpoint_path))
        return checkpoint_path

    @staticmethod
    def __load_model_path(path: str) -> str:
        log.info("Begin loading checkpoint from this file: '{}'".format(path))
        return path

    @staticmethod
    def get_check_point_path(path: str) -> str:
        return (
            ModelSaver.__load_model_path(path)
            if os.path.isfile(path)
            else ModelSaver.__load_model_folder(path)
        )

    @staticmethod
    def load_checkpoint(file_path: str) -> object:
        return paddle.load(file_path)

    @staticmethod
    def load_model(model: object, checkpoint: dict, model_id: int) -> None:
        model_name = 'model_%d' % model_id
        model.set_state_dict(checkpoint[model_name])
        log.info("Model loaded successfully")

    @staticmethod
    def load_opt(opt: object, checkpoint: dict, model_id: int) -> None:
        opt_name = 'opt_%d' % model_id
        opt.set_state_dict(checkpoint[opt_name])
        log.info("opt loaded successfully")

    @staticmethod
    def construct_model_dict(epoch: int, model_list: list, opt_list: list) -> dict:
        assert len(model_list) == len(opt_list)
        model_dict = {'epoch': epoch}
        for i, _ in enumerate(model_list):
            model_name = 'model_%d' % i
            opt_name = 'opt_%d' % i
            model_dict[model_name] = model_list[i].state_dict()
            model_dict[opt_name] = opt_list[i].state_dict()
        return model_dict

    @staticmethod
    def save(file_dir: str, file_name: str, model_dict: dict) -> None:
        paddle.save(model_dict, file_dir + file_name)
        log.info("Save model in : '{}'".format(file_dir + file_name))
        ModelSaver.write_check_point_list(file_dir, file_name)

    @staticmethod
    def write_check_point_list(file_dir: str, file_name: str) -> None:
        fd_checkpoint_list = ModelSaver.__write_check_point_list_title(file_dir)
        if fd_checkpoint_list is None:
            ModelSaver.__write_new_check_point_file(file_dir, file_name)
        else:
            ModelSaver.__write_old_check_point_file(fd_checkpoint_list, file_dir, file_name)


def debug_main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../JackFramework'))
    print(sys.path)
    TEST_FILE_DIR = './Checkpoint/'
    MODEL_FILE_NAME = 'test_model_1_epoch_100.pth'
    ModelSaver.write_check_point_list(TEST_FILE_DIR, MODEL_FILE_NAME)


if __name__ == '__main__':
    debug_main()
