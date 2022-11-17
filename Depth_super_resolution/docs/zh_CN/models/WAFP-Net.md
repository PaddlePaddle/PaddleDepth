# WAFP-Net(IEEE Transactions on Multimedia 2021)

<div align="center">

[English](../../en_US/models/WAFP-Net.md)| [简体中文](DRN.md)

</div>

A paddle implementation of the paper WAFP-Net: Weighted Attention Fusion Based Progressive Residual Learning for Depth Map Super-Resolution,
[\[IEEE Transactions on Multimedia 2021\]](https://ieeexplore.ieee.org/document/9563214)


## 摘要

尽管在深度图超分辨率（DSR）方面取得了显著的进展，但要解决现实世界中低分辨率（LR）深度图的退化问题仍然是一个重大挑战。现有的DSR方法主要使用合成数据集，这与从真实的深度传感器得到的数据有很大不同。此外，现有的DSR方法对特征的增强还不够充分，这也限制了性能。为了缓解这些问题，我们首先提出了两种退化模型来描述LR深度图的生成，包括带噪声的双立方体下采样和间隔下采样，并相应地学习不同的DSR模型。然后，我们提出了一种加权注意力融合策略，该策略被嵌入到渐进残差学习框架中，保证了高分辨率（HR）深度图能够以从粗到细的方式被很好地恢复。加权注意融合策略可以在全局和局部两个方面增强具有丰富高频成分的特征，因此可以预期会有更好的高分辨率深度图。此外，为了充分重用渐进过程中的有效信息，我们在提出的框架中结合了一个多阶段融合模块，并利用总广义变异（TGV）正则化和输入损失来进一步提高我们方法的性能。对不同基准的广泛实验证明了我们的方法比最先进的（SOTA）方法的优越性


## 数据准备

下载2个数据集压缩包：[data_all](docs/zh_CN/datasets/data_all.md)，[DSR-TestData](docs/zh_CN/datasets/DSR-TestData.md) 。解压并将`data_all`文件夹（含133张深度图）和`test_data`文件夹（含4个测试数据）放置成以下位置

```shell
data/
  ├── data_all/
  │   ├── alley_1_1.png
  │   ├── ...
  │   └── ...
  ├── test_data/
  │   ├── cones_x4.mat
  │   ├── teddy_x4.mat
  │   ├── tskuba_x4.mat
  │   └── venus_x4.mat
```

`data_all`中的深度图需要进行下采样，然后将深度相关高斯噪声以θ（d）=N（0，δ/d）的形式添加到低分辨率训练数据中，其中δ=651，d表示DL中每个像素的深度值。使用`matlab`软件，执行`data/process_wafp/generate_train_noise.m`脚本，生成训练数据`train_depth_x4_noise.h5`，注意使用时修改脚本中`data_all`文件夹路径

**注意**：处理好的训练集`train_depth_x4_noise.h5`已上传至AI Studio平台，链接如下：https://aistudio.baidu.com/aistudio/datasetdetail/154330



## 训练

```shell
python -u tools/main.py --config-file configs/wafp_x4.yaml
```


## 测试
**DSR-TestData**

Execute the following command to test the `DSR-TestData` dataset
```shell
python -u tools/main.py --config-file configs/wafp_x4.yaml --evaluate-only --load wafp_x4_best.pdparams
```


## 模型

[Pretraining Model](https://aistudio.baidu.com/aistudio/datasetdetail/176907)
你可以使用这个训练好的权重来重现[README_cn.md](README_cn.md)中报告的结果

## 引用
如果你发现这个代码对你的研究有帮助，请引用:
```
@ARTICLE{9563214,
  author={Song, Xibin and Zhou, Dingfu and Li, Wei and Dai, Yuchao and Liu, Liu and Li, Hongdong and Yang, Ruigang and Zhang, Liangjun},
  journal={IEEE Transactions on Multimedia}, 
  title={WAFP-Net: Weighted Attention Fusion Based Progressive Residual Learning for Depth Map Super-Resolution}, 
  year={2022},
  volume={24},
  pages={4113-4127},
  doi={10.1109/TMM.2021.3118282}}
```