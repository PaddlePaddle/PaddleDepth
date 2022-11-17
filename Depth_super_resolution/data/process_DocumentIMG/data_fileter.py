import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.dirname(cur_path))[0]
sys.path.append(root_path)

import cv2
import numpy as np
import argparse
import tqdm

parser = argparse.ArgumentParser(description="filter images")
parser.add_argument("--x_dir", type=str, default="../data/01/x_sub", help='path to LQ images')
parser.add_argument("--x2_dir", type=str, default="../data/01/x2_sub", help='path to x2 GT images')
parser.add_argument("--x4_dir", type=str, default="../data/01/x4_sub", help='path to x4 GT images')
parser.add_argument("--mode", type=str, default="color", help='path to x4 GT images')

opt = parser.parse_args()

def cal_gradient(img_path):
    img = cv2.imread(img_path)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    return np.mean(sobelxy)

def cal_colormean(img_path):
    img = cv2.imread(img_path, -1)
    img = np.ascontiguousarray(img)
    return np.mean(img)

def check_data():
    inputs_dir = opt.x_dir
    gt_dir_mid = opt.x2_dir
    gt_dir = opt.x4_dir
    mode = opt.mode

    image_list = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    for filename in tqdm.tqdm(image_list):
        gt_path = os.path.join(gt_dir, filename)
        if mode == "color":
            color_mean = cal_colormean(gt_path)
            if color_mean == 255:
                input_path = os.path.join(inputs_dir, filename)
                gt_mid_path = os.path.join(gt_dir_mid, filename)
                os.remove(input_path)
                os.remove(gt_path)
                os.remove(gt_mid_path)
        else:        
            img_gd = cal_gradient(gt_path)
            if img_gd<10:
                input_path = os.path.join(inputs_dir, filename)
                gt_mid_path = os.path.join(gt_dir_mid, filename)
                os.remove(input_path)
                os.remove(gt_path)
                os.remove(gt_mid_path)


if __name__ == "__main__":
    check_data()
