# PaddleStereo: 一个关于双目立体匹配的统一框架
</div>

<div align="center">

[English](README.md)| [简体中文](README_zh-CN.md)

</div>
PaddleStereo 是一款基于 PaddlePaddle 的双目深度估计工具箱，是 PaddleDepth项目的成员之一。
它具有可扩展性，容易上手的特点，此外它在相同的训练策略和环境下公平比较了双目立体匹配领域里面SOTA(state-of-the-art)的算法

## 基准测试和模型库

作为初始版本，PaddldStereo目前支持以下算法。

1. [PCWNet (ECCV2022)[1]](model_document/PCWNet/README.md)
2. [PSMNet (CVPR2018)[2]](model_document/PSMNet/README.md)

请点击上方的超链接查看每个算法的实现细节

## 安装

1.安装[Paddlepaddle](https://www.paddlepaddle.org.cn/install/quick)
 - 版本要求: PaddlePaddle>=2.3.2, Python>=3.8
 - 多GPU版本需要合适的cuda,cudnn,nccl环境. 请在官方网站查看详细的文档说明

2.通过如下命令下载PaddleStereo工具箱

```
git clone https://github.com/PaddlePaddle/PaddleDepth
cd PaddleDepth/Paddlestereo
pip install -r requirements.txt
```

## 数据集准备
你可以参照 [dataset_prepare](data_prepare/data_prepare.md) 来进行数据集的准备.

## 如何使用

1. 在SceneFlow预训练
```shell
$ ./Scripts/start_train_sceneflow_stereo_net_multi.sh
```
2. 在KITTI2012上微调
```shell
$ ./Scripts/start_train_kitti2012_stereo_net_multi.sh
```

### 测试
1.在KITTI2012测试
```shell
$ ./Scripts/start_test_kitti2012_stereo_net.sh
```

## 定制化算法

PaddleStereo的整体代码结构如下:
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

你可以按照如下步骤开发自己的算法:

1. 检查你的模型是否需要模型来进行训练，如果有把模型加入到 `model`中
2. 检查你的模型是否需要新的dataloader，如果有把它们加到`Dataloders`中
3. 加入你自己的启动文件（.sh）

## 结果

我们在KITTI2015以及KITTI2012上评测了paddle stereo已经实现的算法. 

注意我们并没有通过额外的技巧来优化模型结果，因此你可以直接使用.sh的脚本文件来复现我们在表格中报告的精度


### KITTI2015

## 贡献

PaddleStereo工具箱目前还在积极维护与完善过程中。 我们非常欢迎外部开发者为Paddle Depth提供新功能\模型。 如果您有这方面的意愿的话，请往我们的邮箱或者issue里面反馈
## 感谢
PaddleDepth 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。
我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 
我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 参考文献

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



## 联系方式

- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@pku.edu.cn
- [Rao Zhibo](https://github.com/RaoHaocheng): raoxi36@foxmail.com
