# -*- coding: utf-8 -*-
import re
import os
import sys
import random
import linecache
import numpy as np
from PIL import Image
import cv2


class Switch(object):
    def __init__(self, value: str) -> None:
        self.__value = value
        self.__fall = False

    def __iter__(self) -> bool:
        """Return the match method once, then stop"""
        yield self.match
        return StopIteration

    def match(self, *args: tuple) -> bool:
        """Indicate whether to enter a case suite"""
        if self.__fall or not args:
            return True
        elif self.__value in args:  # changed for v1.5, see below
            self.__fall = True
            return True
        else:
            return False


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
        if not os.path.exists(path):
            os.makedirs(path)

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


class DataAugmentation(object):
    """docstring for ClassName"""
    EPSILON = 1e-9

    def __init__(self):
        super().__init__()

    @staticmethod
    def random_org(w: int, h: int, crop_w: int, crop_h: int) -> tuple:
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        return x, y

    @staticmethod
    def standardize(img: object) -> object:
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + DataAugmentation.EPSILON)

    @staticmethod
    def random_crop(imgs: list, w: int, h: int,
                    crop_w: int, crop_h: int) -> list:
        x, y = DataAugmentation.random_org(w, h, crop_w, crop_h)
        imgs = list(map(lambda img: img[y:y + crop_h,
                                        x:x + crop_w, :], imgs))
        return imgs

    @staticmethod
    def random_rotate(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() <= thro:
            rotate_k = np.random.randint(low=0, high=3)
            imgs = list(map(lambda img: np.rot90(img, rotate_k), imgs))
        return imgs

    @staticmethod
    def random_flip(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: np.flip(img, 0), imgs))
        if np.random.random() < thro:
            imgs = list(map(lambda img: np.flip(img, 1), imgs))
        return imgs

    @staticmethod
    def random_horizontal_flip(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[:, ::-1, ...], imgs))
        return imgs

    @staticmethod
    def random_vertical_flip(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[::-1, :, ...], imgs))
        return imgs

    @staticmethod
    def random_rotate90(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img.swapaxes(1, 0)[:, ::-1, ...], imgs))
        return imgs

    @staticmethod
    def random_rotate180(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[:, ::-1, ...][::-1, :, ...], imgs))
        return imgs

    @staticmethod
    def random_rotate270(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img.swapaxes(1, 0)[::-1, :, ...], imgs))
        return imgs

    @staticmethod
    def random_transpose(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img.swapaxes(1, 0), imgs))
        return imgs

    @staticmethod
    def random_transverse(imgs: list, thro: float = 0.5) -> list:
        if np.random.random() < thro:
            imgs = list(map(lambda img: img[:, ::-1, ...].swapaxes(1, 0)[:, ::-1, ...], imgs))
        return imgs


class ImgIO(object):
    """docstring for ImgIO"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def read_img(path: str, io_type=False) -> np.array:
        if io_type:
            img = cv2.imread(path)
        else:
            img = np.array(Image.open(path), np.float32)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        return img

    @staticmethod
    def write_img(path: str, img: np.array) -> None:
        img = np.array(img, np.uint8).squeeze()
        if len(img.shape) == 2 or len(img.shape) == 3 and img.shape[2] == 3:
            img = Image.fromarray(img)
            img.save(path)
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    @staticmethod
    def __get_color(header: str) -> bool:
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        return color

    @staticmethod
    def read_pfm(path: str) -> tuple:
        file = open(path, 'rb')

        header = file.readline().decode('utf-8').rstrip()
        color = ImgIO.__get_color(header)

        dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                             file.readline().decode('utf-8'))

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

    @staticmethod
    def write_pfm(path: str, image: np.array, scale: int = 1) -> None:
        with open(path, mode='wb') as file:
            if image.dtype.name != 'float32':
                raise Exception('Image dtype must be float32.')

            image = np.flipud(image)
            if len(image.shape) == 3 and image.shape[2] == 3:  # color image
                color = True
            elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
                color = False
            else:
                raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

            file.write(str.encode('PF\n' if color else 'Pf\n'))
            file.write(str.encode('%d %d\n' % (image.shape[1], image.shape[0])))

            endian = image.dtype.byteorder
            if endian == '<' or endian == '=' and sys.byteorder == 'little':
                scale = -scale

            file.write(str.encode('%f\n' % scale))

            image_string = image.tostring()
            file.write(image_string)
