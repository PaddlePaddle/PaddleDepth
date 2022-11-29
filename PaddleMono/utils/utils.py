# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
import random
import numpy as np
import paddle
import h5py
from six.moves import urllib

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def load_weight_file(weight_file):
    """
    Load weight from given file
    """
    if os.path.exists(weight_file + '.pdparams'):
        print("loading weight from " + weight_file + '.pdparams')
        weights = paddle.load(weight_file + '.pdparams')
    elif os.path.exists(weight_file + '.h5'):
        print("loading weight from " + weight_file + '.h5')
        weights = {}
        with h5py.File(weight_file + '.h5', 'r') as f:
            for k in f.keys():
                paddle_key = k.replace('running_mean', '_mean')
                paddle_key = paddle_key.replace('running_var', '_variance')
                try:
                    value = paddle.to_tensor(f[k][:])
                except: # handle zero shape
                    value = paddle.to_tensor(f[k][...].item())
                if 'weight' in paddle_key and len(value.shape) == 2:
                    value = value.T
                weights[paddle_key] = value 
    else:
        raise NotImplementedError(f'Do not find validate weight from {weight_file}')
    return weights


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu())
    mi = float(x.min().cpu())
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def gray2rgb(x):
    """
    convert 1 channel gray image to 3 channel rgb image
    (VisualDL is picky about the shape)
    """
    return paddle.expand(x, (3, -1, -1))
    
def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


def format_test_files(file_path, format_file_name='test_files.txt'):
    if not os.path.exists(file_path):
        raise ValueError("The path {} is not exists!".format(file_path))

    file_dir = '/'.join(file_path.split('/')[:-1])
    new_name = format_file_name
    with open(file_path, 'r') as f:
        with open(os.path.join(file_dir, new_name), 'w') as writer:
            for line in f.readlines():
                for img in line.split():
                    ls = img.split('/')
                    n1 = '/'.join(ls[0:2])
                    n2 = ls[4].split('.')[0].lstrip('0')
                    if n2 == '':
                        n2 = '0'
                    n3 = 'l'
                    if ls[2] == 'image_03':
                        n3 = 'r'
                    new_line = n1 + ' ' + n2 + ' ' + n3 + '\n'
                    writer.write(new_line)
