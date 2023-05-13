# -*- coding: utf-8 -*-
import os
import linecache


class FileHandler(object):
    """docstring for FileHandler"""
    ERROR_LINE_NUM = -1

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def mkdir(path: str) -> None:
        # new folder
        path = path.strip()
        path = path.rstrip("\\")
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            pass

    @staticmethod
    def open_file(path: str, is_continue: bool = True) -> object:
        if not is_continue and os.path.exists(path):
            os.remove(path)
        return open(path, 'a+')

    @staticmethod
    def remove_file(path: str) -> None:
        if os.path.isfile(path):
            os.remove(path)

    @staticmethod
    def close_file(fd_file: object) -> None:
        fd_file.close()

    @staticmethod
    def write_file(fd_file: object, data_str: str) -> None:
        data_str = str(data_str)
        fd_file.write(data_str + "\n")
        fd_file.flush()

    @staticmethod
    def get_line(filename: str, line_num: int) -> str:
        path = linecache.getline(filename, line_num)
        path = path.rstrip("\n")
        return path

    @staticmethod
    def insert_str2line(fd_file: object, data_str: str, line_num: int) -> None:
        off_set = FileHandler.__get_line_offset(fd_file, line_num)
        fd_file.seek(off_set)
        FileHandler.write_file(fd_file, data_str)

    @staticmethod
    def get_line_fd(fd_file: object, line_num: int) -> str:
        current_off_set = fd_file.tell()
        fd_file.seek(0, 0)
        if len(fd_file.readlines()) <= line_num:
            return ''
        fd_file.seek(FileHandler.__get_line_offset(fd_file, line_num))
        line = fd_file.readline()
        line = line.rstrip("\n")
        fd_file.seek(current_off_set)
        return line

    @staticmethod
    def copy_file(fd_file_a: object, fd_file_b: object, line_num: int) -> None:
        off_set = FileHandler.__get_line_offset(fd_file_a, line_num)
        fd_file_a.seek(off_set, 0)
        off_set = FileHandler.__get_line_offset(fd_file_b, line_num)
        fd_file_b.seek(off_set, 0)

        next_line = fd_file_a.readline()
        while next_line:
            next_line = next_line.rstrip("\n")
            FileHandler.write_file(fd_file_b, next_line)
            next_line = fd_file_a.readline()

    @staticmethod
    def __get_line_offset(fd_file: object, line_num: int):
        off_set_list = FileHandler.__line2offset(fd_file)
        if line_num > len(off_set_list):
            return FileHandler.ERROR_LINE_NUM
        return off_set_list[line_num]

    @staticmethod
    def __line2offset(fd_file: object) -> list:
        current_off_set = fd_file.tell()
        fd_file.seek(0, 0)
        off_set_list, off_set = [], 0
        off_set_list.append(off_set)

        for line in fd_file:
            off_set += len(line)
            off_set_list.append(off_set)

        fd_file.seek(current_off_set)

        return off_set_list


def debug_main():
    file_dir = './Checkpoint/'
    file_name = 'checkpoint.list'
    fd_checkpoint_list = FileHandler.open_file(file_dir + file_name)
    str_line = FileHandler.get_line_fd(fd_checkpoint_list, 0)
    print(str_line)
    test_file_name = "test_model_1_epoch_%d.pth"

    # Checkpoint's list file
    CHECK_POINT_LIST_NAME = 'checkpoint.list'
    LAST_MODEL_NAME = 'last model name:'

    for i in range(50):
        fd_checkpoint_list = FileHandler.open_file(
            file_dir + CHECK_POINT_LIST_NAME)
        str_line = FileHandler.get_line_fd(fd_checkpoint_list, 0)
        file_name = test_file_name % i
        if str_line[: len(LAST_MODEL_NAME)] != LAST_MODEL_NAME:
            print("yes")
            FileHandler.close_file(fd_checkpoint_list)
            fd_checkpoint_list = None
            os.remove(file_dir + CHECK_POINT_LIST_NAME)

        if fd_checkpoint_list is None:
            fd_checkpoint_list = FileHandler.open_file(
                file_dir + CHECK_POINT_LIST_NAME)
            FileHandler.write_file(
                fd_checkpoint_list, LAST_MODEL_NAME + file_name)
            FileHandler.write_file(fd_checkpoint_list, file_name)
            FileHandler.close_file(fd_checkpoint_list)
        else:
            fd_checkpoint_temp_list = FileHandler.open_file(
                file_dir + CHECK_POINT_LIST_NAME + '.temp')
            FileHandler.write_file(
                fd_checkpoint_temp_list, LAST_MODEL_NAME + file_name)
            FileHandler.copy_file(fd_checkpoint_list,
                                  fd_checkpoint_temp_list, 1)
            FileHandler.write_file(fd_checkpoint_temp_list, file_name)
            FileHandler.close_file(fd_checkpoint_list)
            FileHandler.close_file(fd_checkpoint_temp_list)
            os.remove(file_dir + CHECK_POINT_LIST_NAME)
            os.rename(file_dir + CHECK_POINT_LIST_NAME +
                      '.temp', file_dir + CHECK_POINT_LIST_NAME)


if __name__ == '__main__':
    debug_main()
