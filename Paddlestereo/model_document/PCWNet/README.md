# PCWNet(ECCV 2022)
A paddle implementation of the paper PCW-Net: Pyramid Combination and
Warping Cost Volume for Stereo Matching [\[ECCV 2022\]](https://link.springer.com/content/pdf/10.1007/978-3-031-19824-3_17.pdf)

## Abstract
Existing deep learning based stereo matching methods either
focus on achieving optimal performances on the target dataset while with
poor generalization for other datasets or focus on handling the cross-domain
generalization by suppressing the domain sensitive features which results in
a significant sacrifice on the performance. To tackle these problems, we propose PCW-Net, a Pyramid Combination andWarping cost volume-based
network to achieve good performance on both cross-domain generalization
and stereo matching accuracy on various benchmarks. In particular, our
PCW-Net is designed for two purposes.First, we construct combination volumes on the upper levels of the pyramid and develop a cost volume fusion
module to integrate them for initial disparity estimation.Multi-scale receptive fields can be covered by fusing multi-scale combination volumes, thus,
domain-invariant features can be extracted. Second, we construct the warping volume at the last level of the pyramid for disparity refinement. The proposed warping volume can narrow down the residue searching range from the
initial disparity searching range to a fine-grained one, which can dramatically alleviate the difficulty of the network to find the correct residue in an
unconstrained residue searching space.When training on synthetic datasets
and generalizing to unseen real datasets, our method shows strong crossdomain generalization and outperforms existing state-of-the-arts with a
large margin. After fine-tuning on the real datasets, our method ranks 1st
on KITTI 2012, 2nd on KITTI 2015, and 1st on the Argoverse among all
published methods as of 7, March 2022.

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
**Note**: Plase update .csv file in `Datasets/Stereo` and choose `--modelName PCWNet`

## Models

Models will be open-sourced later

## Link
we also provide the official pytorch implementation in this [website](https://github.com/gallenszl/PCWNet)

## Citation
If you find this code useful in your research, please cite:
```
@inproceedings{shen2022pcw,
  title={PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching},
  author={Shen, Zhelun and Dai, Yuchao and Song, Xibin and Rao, Zhibo and Zhou, Dingfu and Zhang, Liangjun},
  booktitle={European Conference on Computer Vision},
  pages={280--297},
  year={2022},
  organization={Springer}
}
```
