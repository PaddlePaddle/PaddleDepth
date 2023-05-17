import os
from glob import glob
import argparse
import pandas as pd

def generate_kitti_list(args):
    left_images_list = sorted(glob(os.path.abspath(os.path.join(args.left_dir, '*_10.png'))))
    right_images_list = sorted(glob(os.path.abspath(os.path.join(args.right_dir, '*_10.png'))))
    disp_images_list = sorted(glob(os.path.abspath(os.path.join(args.disp_dir, '*_10.png'))))
    dataframe = pd.DataFrame({'left_img': left_images_list,'right_img': right_images_list, 'gt_disp': disp_images_list})

    dataframe.to_csv(args.save_path, index=False, sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir', help="The directory of left images", type=str, default=None)
    parser.add_argument('--right_dir', help="The directory of right images", type=str, default=None)
    parser.add_argument('--disp_dir', help="The directory of disp images", type=str, default=None)
    parser.add_argument('--save_path', help="The path of list file, eg: kitti2012_training_list.csv", type=str, default=None)
    
    args = parser.parse_args()
    generate_kitti_list(args)