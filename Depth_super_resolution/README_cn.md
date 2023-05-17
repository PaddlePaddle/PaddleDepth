# Paddle-DSR：飞桨深度图超分辨开发套件  
</div>

<div align="center">

[English](README.md)| [简体中文](README_cn.md)

</div>

Paddle-DSR 是一款基于 PaddlePaddle 的**深度图超分辨**开发套件，是 PaddleDepth项目的成员之一。 它具有可扩展性，容易上手的特点，此外它在相同的数据集和环境下，以与原论文相符的训练配置，公平比较了深度图超分辨率领域以及图像超分辨率里SOTA(state-of-the-art)的算法

| cones| tskuba | teddy | venus |
| --- | --- | --- | ---|
| ![](https://ai-studio-static-online.cdn.bcebos.com/c16beee3e7c94284ae4e4b80f1f493af4477ef019b2a4efd9cb0c604b36be866)| ![](https://ai-studio-static-online.cdn.bcebos.com/9ccf5207aa1d4285b4f57c66bb5ae47b086c3df2d74d4c54b100b8d79e68f411)| ![](https://ai-studio-static-online.cdn.bcebos.com/ca98f5eb54ba4a0c8a275bd4afdd0c1ef45ac4e70d484762b0ad93745290d426)|![](https://ai-studio-static-online.cdn.bcebos.com/3137984e2b2342139e1dbaf78ab8abc49c869340f19743e7b804d632129cd413) |


## 基准测试和模型库

作为初始版本，Paddle-DSR-Lab目前支持以下算法（点击下方超链接查看各个算法的详细使用教程）。
1. [WAFP-Net (IEEE Transactions on Multimedia 2021)[1]](docs/zh_CN/models/WAFP-Net.md)
2. [PMBANet (IEEE Transactions on Image Processing 2019)[2]](docs/zh_CN/models/PMBANet.md)
3. [RCAN (ECCV 2018)[3]](docs/zh_CN/models/RCAN.md)
4. [DRN (CVPR 2020)[4]](docs/zh_CN/models/DRN.md)


## 安装

你可以根据如下步骤安装Paddle-DSR工具箱：

- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
    - 版本要求：PaddlePaddle>=2.2.0, Python>=3.7

- Paddle-DSR-Lab安装，通过以下命令
```
git clone https://github.com/PaddlePaddle/PaddleDepth.git
cd Depth_super_resolution
pip install -r requirements.txt
```

## 数据集准备

你可以参照[数据集准备](docs/zh_CN/datasets)文件夹下的[训练集](docs/zh_CN/datasets/data_all.md)和[测试集](docs/zh_CN/datasets/DSR-TestData.md)数据准备文档来进行相关模型的数据集准备

## 如何使用

### 训练模型

```shell
python -u tools/main.py --config-file $file_path$
```

- `config-file`参数为训练模型的配置文件路径
- 若有预训练权重进行finetune，则运行以下命令启动训练，`load`参数为预训练权重的路径

```shell
python -u tools/main.py --config-file $file_path$ --load $weight_path$
```

- 若训练中断，需要恢复训练，则运行以下命令，`resume`参数为checkpoint路径

```shell
python -u tools/main.py --config-file $file_path$ --resume $checkpoint_path$
```


### 测试模型

```shell
python -u tools/main.py --config-file $file_path$ --evaluate-only --load $weight_path$
```

## 定制化算法

Paddle-DSR工具箱的文件结构如下所示

```shell
Depth_super_resolution
    │  README.md                # 英文说明文档
    │  README_cn.md             # 中文说明文档
    │  requirements.txt         # 安装依赖文件
    ├─configs                   # 配置文件
    ├─data                      # 数据处理
    │  ├─process_DocumentIMG    # 处理百度网盘超分比赛数据
    │  ├─process_pmba           # 处理PMBA所用数据
    │  └─process_wafp           # 处理WAFP所用数据
    ├─docs                      # 模型以及数据集的介绍文档
    ├─ppdsr 
    │  ├─datasets               # 数据类定义、加载相关
    │  │  └─preprocess          # 数据增强相关
    │  ├─engine                 # 训练、测试总体代码
    │  ├─metrics                # 评估指标相关
    │  ├─models                 # 模型相关
    │  │  ├─backbones           # 可共用的模型backbone
    │  │  ├─criterions          # 损失函数
    │  │  └─generators          # 模型组网文件
    │  ├─modules                # 组网辅助文件，如参数初始化等
    │  ├─solver                 # 优化器相关
    │  └─utils                  # 数据读取、日志显示等辅助工具
    └─tools                     # 训练、测试启动工具
```

你可以按照如下步骤开发自己的算法:

1. 检查你的模型是否需要新的损失函数来进行训练，如果有把损失函数加入到 `ppdsr/models/criterions`中
2. 检查你是否需要新增模型来进行训练，如果有把模型加入到 `ppdsr/models`中
3. 检查你是否需要新增数据集处理方式来进行训练，如果有把数据集加入到 `ppdsr/datasets`中
4. 在`configs`中，加入你自己的配置文件（.yaml）


## 结果

我们在使用`teddy`、`cones`、`tskuba`、`venus`四张深度图作为测试集`DSR-TestData`，评测了Paddle-DSR已经实现的算法. 

**注意**: 我们并没有通过额外的技巧来优化模型结果，因此你可以直接使用.yaml的配置文件来复现我们在表格中报告的精度

### DSR-TestData
|     Model        | PSNR | SSIM | RMSE | MAD | size  | 
|-------------|-------|-------|-------|-------|--------|
| WAFP-Net [1]| 42.0344 | 0.9834 | 2.5561 | 0.9246 | 3M | 
| PMBANet [2] | 41.0418 | 0.9825 | 2.4728 | 0.6278 | 94.9M  |
| RCAN [3]    | 42.5297 | 0.9850 | 2.4401 | 0.6685 | 59.6M  | 
| DRN [4]     | 42.4906 | 0.9850 | 2.4634 | 0.6506 | 18.4M  | 


## 贡献

Paddle-DSR工具箱目前还在积极维护与完善过程中。 我们非常欢迎外部开发者为Paddle-DSR-Lab提供新功能\模型。 如果您有这方面的意愿的话，请往我们的邮箱或者issue里面反馈

## 参考文献

[1] Song, Xibin, Dingfu Zhou, Wei Li, Yuchao Dai, Liu Liu, Hongdong Li, Ruigang Yang, and Liangjun Zhang. ‘WAFP-Net: Weighted Attention Fusion Based Progressive Residual Learning for Depth Map Super-Resolution’. IEEE Transactions on Multimedia 24 (2022): 4113–27. https://doi.org/10.1109/TMM.2021.3118282.
.

[2] Ye, Xinchen, Baoli Sun, Zhihui Wang, Jingyu Yang, Rui Xu, Haojie Li, and Baopu Li. ‘PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution’. IEEE Transactions on Image Processing 29 (2020): 7427–42. https://doi.org/10.1109/TIP.2020.3002664.

[3] Zhang, Yulun, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. ‘Image Super-Resolution Using Very Deep Residual Channel Attention Networks’. In Computer Vision – ECCV 2018, edited by Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, 11211:294–310. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2018. https://doi.org/10.1007/978-3-030-01234-2_18.


[4] Guo, Yong, Jian Chen, Jingdong Wang, Qi Chen, Jiezhang Cao, Zeshuai Deng, Yanwu Xu, and Mingkui Tan. ‘Closed-Loop Matters: Dual Regression Networks for Single Image Super-Resolution’. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5406–15. Seattle, WA, USA: IEEE, 2020. https://doi.org/10.1109/CVPR42600.2020.00545.


## 联系方式

- [Yuanhang Kong](https://github.com/kongdebug): KeyK@foxmail.com
- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@baidu.com


