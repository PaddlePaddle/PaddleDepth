# RAFT-Stereo(3DV 2021)

A paddle implementation of the paper RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching [\[3DV 2021\]](https://arxiv.org/pdf/2109.07547.pdf)

## Abstract

RAFT-Stereo is a new deep architecture for rectified stereo based on the optical flow network RAFT. RAFT-Stereo introduce multi-level convolutional GRUs, which more efficiently propagate information across the image. A modified version of RAFT-Stereo can perform accurate real-time inference. RAFT-stereo ranks first on the Middlebury leaderboard, outperforming the next
best method on 1px error by 29% and outperforms all published work on the ETH3D two-view stereo benchmark.

### Train

We did not train `RAFT-Stereo` on the `Sceneflow` dataset, but instead converted the [original paper repo](https://github.com/princeton-vl/RAFT-Stereo) provided weights into `. pdparams` format weights. We provide the checkpoint files [baidu wangpan](https://pan.baidu.com/s/1SpWlQRLyJCeZw3jiPhprSA?pwd=h9hl) that needs  to be downloaded and placed in the `Paddlestereo` folder.

1. Fine-tuning (KITTI 2015)

```shell
$ ./Scripts/start_train_kitti2015_raft_stereo_multi.sh
```
**Note**: Plase update .csv file in `Datasets/Stereo`, update output floder `--outputDir` and update checkpoint file path `--modelDir`.

### Test

1. KITTI2015

```shell
$ ./Scripts/start_test_kitti2015_raftstereo.sh
```
**Note**: Plase update .csv file in `Datasets/Stereo`, update checkpoint file path `--modelDir`.

## Model

We provide checkpoint files [raft_kitti2015](https://pan.baidu.com/s/1E6uH8sJDgJjmxqORWLtBLg?pwd=v8i4) that finetuning on the KITTI2015 dataset. Submit the test results to the KITTI official website for evaluation, with the accuracy shown in the table below:

| Error |D1-bg | D1-fg | D1-all |
| --- | --- | --- | --- |
| All / All | 1.67 | 2.72 | 1.85 |
| All / Est | 1.67 | 2.72 | 1.85 |
| Noc / All | 1.53 | 2.56 | 1.70 |
| Noc / Est | 1.53 | 2.56 | 1.70 |

## Citation
If you find this code useful in your research, please cite:
```
@inproceedings{lipson2021raft,
  title={RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching},
  author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```