# Prepare NYU Depth V2 dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
```

This project use the preprocessed [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset in HDF5 formats. You can 
download [here](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz). This dataset requires 32G of storage space.

You can download the dataset by running the following command in project root directory:

```bash
sh data_prepare/NYU_Depth/download_nyu.sh
```

and the folder structure should be like this:

```text
data
|   nyudepth_hdf5
|   |   |   ├── train
|   |   |   |   ├── basement_0001a
|   |   |   |   |   ├── xxxx.h5
|   |   |   |   ├── basement_0001b
|   |   |   |   |   ├── xxxx.h5
|   |   |   ├── val
|   |   |   |   ├── official
|   |   |   |   |   ├── xxxx.h5
|   |   |   ├── train.csv
|   |   |   ├── val.csv
```