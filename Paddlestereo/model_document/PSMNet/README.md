# PSMNet(CVPR 2018)
A paddle implementation of the paper Pyramid Stereo Matching Network [\[CVPR 2018\]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf)

## Abstract
Recent work has shown that depth estimation from a
stereo pair of images can be formulated as a supervised
learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on
patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in ill-posed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two
main modules: spatial pyramid pooling and 3D CNN. The
spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume.
The 3D CNN learns to regularize cost volume using stacked
multiple hourglass networks in conjunction with intermediate supervision. The proposed approach was evaluated
on several benchmark datasets. Our method ranked first in
the KITTI 2012 and 2015 leaderboards before March 18,2018.


### Train
1. Pre-training
```shell
$ ./Scripts/start_train_sceneflow_stereo_net_multi.sh
```
2. Fine-tuning (KITTI 2012)
```shell
$ ./Scripts/start_train_kitti2012_stereo_net_multi.sh
```

### Test
1. KITTI2012
```shell
$ ./Scripts/start_test_kitti2012_stereo_net.sh
```
**Note**: Plase update .csv file in `Datasets/Stereo` and choose `--modelName PSMNet`

## Models

Models will be open-sourced later

## Citation
If you find this code useful in your research, please cite:
```
@inproceedings{chang2018pyramid,
  title={Pyramid stereo matching network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5410--5418},
  year={2018}
}
```