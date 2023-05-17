# PaddleCompletion: 一个关于深度信息补全的统一框架

<div align="center">

[English](README.md)| [简体中文](README_zh-CN.md)

</div>
PaddleCompletion 是一款基于 PaddlePaddle 的深度信息补全工具箱，是 PaddleDepth 项目的成员之一。
它具有可扩展性，容易上手的特点，此外它在相同的训练策略和环境下公平比较了深度信息补全领域里面SOTA(state-of-the-art)的算法。

## 基准测试和模型库

作为初始版本，PaddleCompletion目前支持以下算法。


1. [CSPN (ECCV2018)](model_document/CSPN/README.md)
2. [FCFRNet (AAAI2021)](model_document/FCFRNet/README.md)
3. [STD (ICRA2019)](model_document/STD/README.md)
4. [GuideNet (IEEE Transactions on Image Processing)](model_document/GuideNet/README.md)

请点击上方的超链接查看每个算法的实现细节

## 安装

你可以通过如下命令下载 PaddleCompletion 工具箱

```
git clone https://github.com/PaddlePaddle/PaddleDepth
cd PaddleDepth/PaddleCompletion
pip install -r requirements.txt
```

PaddleCompletion 基于 PaddlePaddle 2.3.2 版本开发，请使用 python 3.9 运行 PaddleCompletion 。

## 数据集准备

你可以参照 [dataset_prepare](data_prepare/data_prepare.md) 来进行数据集的准备。

## 如何使用

### 训练

1. 修改`config`文件夹下对应模型的 .yaml 文件。
2. 指定confing文件运行 `train.py`,例如： `python train.py -c configs/CSPN.yaml`

* 也可以通过我们提供的脚本运行来复现我们的结果: `bash scripts/train_cspn.sh`.

### 测试

1. 修改`config`文件夹下对应模型的 .yaml 文件。
2. 下载我们提供的预训练模型，放置对应的目录下，如：`weights/cspn/model_best.pdparams`。
3. 指定confing文件运行 `test.py`,例如： `python evaluate.py -c configs/CSPN.yaml`。

## 定制化算法

你可以按照如下步骤开发自己的算法:

1. 检查你的模型是否需要新的损失函数来进行训练，如果有把损失函数加入到 `loss_funcs`中
2. 检查你的模型是否需要模型来进行训练，如果有把模型加入到 `model`中
3. 加入你自己的配置文件（.sh or .yaml）

## 结果

我们在KITTI2015以及NYU Depth V2上评测了 PaddleCompletion 已经实现的算法。
注意我们并没有通过额外的技巧来优化模型结果，因此你可以直接使用.sh的脚本文件来复现我们在表格中报告的精度。

### KITTI

| Method    | RMSE    | MAE     | iRMSE | iMAE  |
|-----------| ------- | ------- | ----- | ----- |
| `FCFRNet` | 784.224 | 222.639 | 2.370 | 1.014 |
| `STD` | 814.73 | 242.639 | 2.80 | 1.21 |
| `GuideNet` | 745.41 | 227.95 | 2.48 | 1.09 |

### NYU Depth V2

| Data            | RMSE   | REL    | DELTA1.02 | DELTA1.05 | DELTA1.10 |
|-----------------| ------ | ------ | --------- | --------- | --------- |
| `CSPN`          | 0.1111 | 0.0151 | 0.8416    | 0.9386    | 0.9729    |


## 贡献

PaddleCompletion 工具箱目前还在积极维护与完善过程中。 我们非常欢迎外部开发者为Paddle Depth提供新功能\模型。 如果您有这方面的意愿的话，请往我们的邮箱或者issue里面反馈

## 感谢

PaddleDepth 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。
我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 
我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 参考文献

[1] [CSPN: A Compact Spatial Propagation Network for Depth Completion](https://openaccess.thecvf.com/content_ECCV_2018/html/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.html)

[2] [FCFRNet: Fast and Convergent Feature Refinement Network for Depth Completion](https://doi.org/10.1609/aaai.v35i3.16311)

[3] [STD: Self-Supervised Sparse-to-Dense: Self-Supervised Depth Completion from LiDAR and Monocular Camera](https://arxiv.org/pdf/1807.00275.pdf)

[4] [GuideNet: Learning guided convolutional network for depth completion](https://arxiv.org/abs/1908.01238)

[comment]: <> (## Citation)

[comment]: <> (If you think this toolkit or the results are helpful to you and your research, please cite us!)

[comment]: <> (```)

[comment]: <> (@Misc{deepda,)

[comment]: <> (howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   )

[comment]: <> (title = {DeepDA: Deep Domain Adaptation Toolkit},  )

[comment]: <> (author = {Wang, Jindong and Hou, Wenxin})

[comment]: <> (}  )

[comment]: <> (```)



## 联系方式

- [Juntao Lu](https://github.com/ralph0813): juntao.lu@student.unimelb.edu.au
- [Bing Xiong](https://github.com/imexb9584): bingxiong9527@siat.edu.cn
- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@pku.edu.cn
