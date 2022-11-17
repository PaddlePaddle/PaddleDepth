import os
import shutil

import argparse

parser = argparse.ArgumentParser(description="combine images")
parser.add_argument("--root_dir", type=str, default="../data/", help='path to root')
parser.add_argument("--save_root", type=str, default="./data/DocIMG/", help='path to save root')
opt = parser.parse_args()

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def combine_data(root_dir, save_root):
    make_dir(save_root)
    save_x_dir = os.path.join(save_root, 'x')
    make_dir(save_x_dir)
    save_x2_dir = os.path.join(save_root, 'x2')
    make_dir(save_x2_dir)
    save_x4_dir = os.path.join(save_root, 'x4')
    make_dir(save_x4_dir)

    for dirs in os.listdir(root_dir):
        if dirs[0]== ".":
            continue
        sub_dir = os.path.join(root_dir, dirs) 
        print(sub_dir)
        x_dir = os.path.join(sub_dir, 'x_sub')
        x2_dir = os.path.join(sub_dir, 'x2_sub')
        x4_dir = os.path.join(sub_dir, 'x4_sub')
        file_list = [f for f in os.listdir(x_dir) if f.endswith('.png')]
        for filename in file_list:
            x_path = os.path.join(x_dir, filename)
            new_x_path = os.path.join(save_x_dir, filename)
            x2_path = os.path.join(x2_dir, filename)
            new_x2_path = os.path.join(save_x2_dir, filename)
            x4_path = os.path.join(x4_dir, filename)
            new_x4_path = os.path.join(save_x4_dir, filename)

            shutil.move(x_path, new_x_path)
            shutil.move(x2_path, new_x2_path)
            shutil.move(x4_path, new_x4_path)


if __name__ == "__main__":
    root_dir = opt.root_dir
    save_root = opt.save_root
    combine_data(root_dir, save_root)