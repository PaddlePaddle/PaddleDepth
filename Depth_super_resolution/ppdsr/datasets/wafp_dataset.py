#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import os

try:
    import h5py
except ImportError as e:
    print(
        f"{e}, [h5py] package and it's dependencies is required for WAFP-Net.")
try:
    import scipy.io as sio
except ImportError as e:
    print(
        f"{e}, [scipy] package and it's dependencies is required for WAFP-Net.")
import numpy as np

from paddle.io import Dataset

from .builder import DATASETS


def is_mat_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ['mat'])


@DATASETS.register()
class HDF5Dataset(Dataset):
    """Decode HDF5 data using h5py

    Args:
        BaseDataset (_type_): _description_
    """
    def __init__(self,
                 file_path):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        # NOTE: load hdf5 data in memory.
        self.load_file()

    def load_file(self):
        """Load index file to get h5 information."""
        hf = h5py.File(self.file_path, mode="r")
        print("HDF5 Data Loaded from {}".format(self.file_path))
        self.data = hf.get('data')
        self.target = hf.get('label')
        if self.data.shape[0] != self.target.shape[0]:
            raise ValueError(
                f"number of input data({self.data.shape[0]}) "
                f"must equals to label data({self.target.shape[0]})")
        # keep it open, or error will occurs
        # hf.close()
        self.sizex = self.data.shape[0]  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        idx = index % self.sizex
        lr = copy.deepcopy(self.data[idx])
        hr = copy.deepcopy(self.target[idx])
        results = {'imgs': lr, 'labels': hr}
        return np.array(results['imgs']), np.array(results['labels'])


@DATASETS.register()
class MATDataset(Dataset):
    def __init__(self,
                 file_path):
        super(MATDataset, self).__init__()
        data_files = sorted(os.listdir(file_path))
        self.data_filenames = [
            os.path.join(file_path, x) for x in data_files
            if is_mat_file(x)
        ]
        self.sizex = len(self.data_filenames)  # get the size of target


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        """Prepare the frames for training/valid given index. """
        idx = index % self.sizex
        filepath = self.data_filenames[idx]

        mat = sio.loadmat(filepath)
        im_gt_y = mat.get('im_gt_y', None)  # [h, w]
        im_b_y = mat.get('im_b_y', None)  # [h, w]
        if im_gt_y is not None:
            im_gt_y = im_gt_y.astype('float32')
            im_gt_y = np.expand_dims(im_gt_y, axis=0)
        if im_b_y is not None:
            im_b_y = im_b_y.astype('float32')
            im_b_y = np.expand_dims(im_b_y, axis=0)

        results = {}
        results['imgs'] = im_b_y
        results['labels'] = im_gt_y
        
        return results['imgs'], results['labels']


        

    

    

