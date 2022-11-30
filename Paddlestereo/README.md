# PaddleStereo: A Unified Framework for Stereo Matching
</div>

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md)

</div>
A lightweight, easy-to-extend, easy-to-learn, high-performance, and for-fair-comparison toolkit based 
on PaddlePaddle for Stereo Matching. It is a part of the Paddledepth project.


## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

1. [PCWNet (ECCV2022)[1]](model_document/PCWNet/README.md)
2. [PSMNet (CVPR2018)[2]](model_document/PSMNet/README.md)

Please click the hyperlink of each algorithm for more detailed explanation.

## Installation

You can either git clone this whole repo by:
- [Install PaddlePaddle ](https://www.paddlepaddle.org.cn/install/quick)
    - Version requirements: PaddlePaddle>=2.3.2, Python>=3.8
    - Multi-GPU Version need suitable cuda,cudnn,nccl version. see the official website for more detailed explanation.

```
git clone https://github.com/PaddlePaddle/PaddleDepth
cd PaddleDepth/Paddlestereo
pip install -r requirements.txt
```

## Dataset
see guidance in [dataset_prepare](data_prepare/data_prepare.md) for dataset preparation.

## Usage
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

## Customization

The file structure of PaddleStereo is shown below:
```shell
PaddleStereo
    │ README.md 
    │ requirements.txt
    │- Datasets
    │- Example
    │- Scripts
    │- Source
    │  │- Algorithm
    │  │- Core
    │  │- FileHandler
    │  │- ImgHandler
    │  │- SysBasic
    │  │- Template
    │  │- Tools
    │  └─ UserModelImplementation
    │     │- Dataloaders
    │     └─ Models
```

It is easy to design your own method following the 3 steps:

1. Check and write your own model's to `Models`
2. Check and write your own dataloder's to `Dataloders`
3. Write your own start file (.sh)

## Results

We present results of our implementations on 2 popular benchmarks: KITTI2015 and KITTI2012. 
We did not perform careful parameter tuning and simply used the default config files. 
You can easily reproduce our results using provided shell scripts!


### KITTI2015

[comment]: <> (|     Method        | D - A | D - W | A - W | W - A | A - D  | W - D  | Average |)

[comment]: <> (|-------------|-------|-------|-------|-------|--------|--------|---------|)

[comment]: <> (| Source-only | 66.17 | 97.61 | 80.63 | 65.07 | 82.73  | 100.00 | 82.03   |)

[comment]: <> (| DAN [2] &#40;DDC [1]&#41;        | 68.16 | 97.48 | 85.79 | 66.56 | 84.34  | 100.00 | 83.72   |)

[comment]: <> (| DeepCoral [3]       | 66.06 | 97.36 | 80.25 | 65.32 | 82.53  | 100.00 | 81.92   |)

[comment]: <> (| DANN [4]        | 67.06 | 97.86 | 84.65 | 71.03 | 82.73  | 100.00 | 83.89   |)

[comment]: <> (| DSAN [5]        | 76.04 | 98.49 | 94.34 | 72.91 | 89.96  | 100.00 | 88.62   |)

[comment]: <> (| BNM [7]        | 72.38 | 98.62 | 86.04 | 66.56 | 86.55  | 100.00 |  85.02  |)


[comment]: <> (### Office-Home)

[comment]: <> (|     Method       | A - C | A - P | A - R | C - A | C - P | C - R | P - A | P - C | P - R | R - A | R - C | R - P | Average |)

[comment]: <> (|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|)

[comment]: <> (| Source-only | 51.04 | 68.21 | 74.85 | 54.22 | 63.64 | 66.84 | 53.65 | 45.41 | 74.57 | 65.68 | 53.56 | 79.34 | 62.58   |)

[comment]: <> (| DAN [2] &#40;DDC [1]&#41;       | 52.51 | 68.48 | 74.82 | 57.48 | 65.71 | 67.82 | 55.42 | 47.51 | 75.28 | 66.54 | 54.36 | 79.91 | 63.82   |)

[comment]: <> (| DeepCoral [3]      | 52.26 | 67.72 | 74.91 | 56.20 | 64.70 | 67.48 | 55.79 | 47.17 | 74.89 | 66.13 | 54.34 | 79.05 | 63.39   |)

[comment]: <> (| DANN [4]        | 51.48 | 67.27 | 74.18 | 53.23 | 65.10 | 65.41 | 53.15 | 50.22 | 75.05 | 65.35 | 57.48 | 79.45 | 63.12   |)

[comment]: <> (| DSAN [5]        | 54.48 | 71.12 | 75.37 | 60.53 | 70.92 | 68.53 | 62.71 | 56.04 | 78.29 | 74.37 | 60.34 | 82.99 | 67.97   |)

[comment]: <> (| BNM [7]        | 53.33 | 70.40 | 76.89 | 60.90 | 71.55 | 72.07 | 60.65 | 49.90 | 78.66 | 69.51 | 57.30 | 81.01 | 66.85   |)


## Contribution

The toolkit is under active development and contributions are welcome! 
Feel free to submit issues or emails to ask questions or contribute your code. 
If you would like to implement new features, please submit a issue or emails to discuss with us first.

## Acknowledgement
PaddleDepth is an open source project that is contributed by researchers and engineers 
from various colleges and companies. 
We appreciate all the contributors who implement their methods or add new features, 
as well as users who give valuable feedbacks. 
We wish that the toolbox and benchmark could serve the growing research community by 
providing a flexible toolkit to reimplement existing methods and develop their new algorithms.

## References

[1] Shen, Zhelun, et al. "PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching." European Conference on Computer Vision. 2022.

[2] Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[comment]: <> (## Citation)

[comment]: <> (If you think this toolkit or the results are helpful to you and your research, please cite us!)

[comment]: <> (```)

[comment]: <> (@Misc{deepda,)

[comment]: <> (howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   )

[comment]: <> (title = {DeepDA: Deep Domain Adaptation Toolkit},  )

[comment]: <> (author = {Wang, Jindong and Hou, Wenxin})

[comment]: <> (}  )

[comment]: <> (```)



## Contact

- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@pku.edu.cn
- [Rao Zhibo](https://github.com/RaoHaocheng): raoxi36@foxmail.com
