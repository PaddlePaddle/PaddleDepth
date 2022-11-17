import cv2
import os
import scipy.io as sio
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="filter images")
parser.add_argument("--mat_dir", type=str, default= r"./data/test_data/", help='path to Mat Data')
parser.add_argument("--save_hr_dir", type=str, default=r"./data/PMBA/test_HR/", help='path to save HR images')
parser.add_argument("--save_lq_dir", type=str, default=r"./data/PMBA/test_LR_x4/", help='path to x4 GT images')
parser.add_argument("--scale", type=int, default=4, help='scale')

opt = parser.parse_args()

def generate_test_data(mat_dir, save_hr_dir, save_lq_dir, scale):
    if not os.path.isdir(save_hr_dir):
        os.mkdir(save_hr_dir)
    if not os.path.isdir(save_lq_dir):
        os.mkdir(save_lq_dir)

    filelist = os.listdir(mat_dir)

    for filename in filelist:
        filepath = os.path.join(mat_dir, filename) # 单个的mat文件
        mat = sio.loadmat(filepath)
        im_gt_y = mat.get('im_gt_y', None)  # [h, w]
        im_b_y = mat.get('im_b_y', None)  # [h, w]
        if im_gt_y is not None:
            im_gt_y = im_gt_y.astype('float32')
            im_gt_y = (im_gt_y * 255.).round()
        if im_b_y is not None:
            im_b_y = im_b_y.astype('float32')
            h, w = im_b_y.shape[:2]
            im_b_y = cv2.resize(im_b_y, (int(w // scale), int(h // scale)), cv2.INTER_CUBIC)
            im_b_y = (im_b_y * 255.).round()
        
        im_gt_y = im_gt_y.astype(np.uint8)
        im_b_y = im_b_y.astype(np.uint8)
        name, _ = os.path.splitext(filename)
        cv2.imwrite(os.path.join(save_hr_dir, name+'.png'), im_gt_y)
        cv2.imwrite(os.path.join(save_lq_dir, name+'.png'), im_b_y)


if __name__ == "__main__":
    mat_dir = opt.mat_dir
    save_hr_dir = opt.save_hr_dir
    save_lq_dir = opt.save_lq_dir
    scale = opt.scale

    generate_test_data(mat_dir, save_hr_dir, save_lq_dir, scale)