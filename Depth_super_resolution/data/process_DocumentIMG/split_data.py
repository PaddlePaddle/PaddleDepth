import os
from random import shuffle
import shutil

import argparse

parser = argparse.ArgumentParser(description="combine images")
parser.add_argument("--source", type=str, default="./data/DocIMG/", help='path to source')
parser.add_argument("--target", type=str, default="./data/test_Doc/", help='path to target')
parser.add_argument("--num", type=int, default=500, help='val images num')
opt = parser.parse_args()

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def split_data(source, target, num):
    make_dir(target)
    val_x = os.path.join(target, 'x')
    make_dir(val_x)
    val_x2 = os.path.join(target, 'x2')
    make_dir(val_x2)
    val_x4 = os.path.join(target, 'x4')
    make_dir(val_x4)

    source_x = os.path.join(source, 'x')
    source_x2 = os.path.join(source, 'x2')
    source_x4 = os.path.join(source, 'x4')

    file_list = [f for f in os.listdir(source_x) if f.endswith('.png')]
    shuffle(file_list)

    for i, filename in enumerate(file_list):
        x_path = os.path.join(source_x, filename)
        new_x_path = os.path.join(val_x, filename)
        x2_path = os.path.join(source_x2, filename)
        new_x2_path = os.path.join(val_x2, filename)
        x4_path = os.path.join(source_x4, filename)
        new_x4_path = os.path.join(val_x4, filename)

        shutil.move(x_path, new_x_path)
        shutil.move(x2_path, new_x2_path)
        shutil.move(x4_path, new_x4_path)
        if i == num:
            break

if __name__ == "__main__":
    source = opt.source
    target = opt.target
    num = opt.num
    split_data(source, target, num)