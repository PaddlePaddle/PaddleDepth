# KITTI raw dataset

## Training data
You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

The structure of the folder after unzip:

```text
|-- data/kitti_data
	|-- 2011_10_03
		|-- 2011_10_03_drive_0027_sync
			|-- image_00
			|-- image_01
			|-- image_02
			|-- ...
		|-- 2011_10_03_drive_0034_sync
			|-- ......
		|-- 2011_10_03_drive_0042_sync
			|-- ......
		|-- 2011_10_03_drive_0047_sync
		    |-- ......
		|-- calib_cam_to_cam.txt
		|-- calib_imu_to_velo.txt
		|-- calib_velo_to_cam.txt
```

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

The above conversion command creates images which match our experiments, where KITTI `.png` images were converted to `.jpg` on Ubuntu 16.04 with default chroma subsampling `2x2,1x1,1x1`.
We found that Ubuntu 18.04 defaults to `2x2,2x2,2x2`, which gives different results, hence the explicit parameter in the conversion command.

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


## Testing data

Please download the GT depth and put it in `PaddleMono/splits/eigen/` directory before executing the evaluation script.
[GT_depth](https://drive.google.com/file/d/1D94FFJo2vf6obW4M6PfvSoYRh8a9nD9w/view?usp=sharing)

## Custom dataset

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.
