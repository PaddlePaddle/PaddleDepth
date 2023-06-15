# PaddleMono: 一个关于单目深度估计的统一框架
</div>

<div align="center">

[English](README.md)| [简体中文](README_zh-CN.md)

</div>
PaddleMono是一款基于 PaddlePaddle 的单目深度估计工具箱，是 PaddleDepth项目的成员之一。
它具有可扩展性，容易上手的特点，此外它在相同的训练策略和环境下公平比较了单目深度估计领域里面SOTA(state-of-the-art)的算法

## 基准测试和模型库

作为初始版本，PaddldMono目前支持以下算法。

[comment]: <> (- Monodepth2)

[comment]: <> (- MLDA-Net)

[comment]: <> (- Depth Hints &#40;以上两个模型训练时均可开启Depth Hints&#41;)

[comment]: <> (- BTS)


1. [Monodepth2 (ICCV2019)[1]](configs/monodepthv2/README.md)
2. [MLDA-Net (TIP2021)[2]](configs/mldanet/README.md)
3. [Depth Hints (ICCV2019)[3]](configs/depth_hints/README.md)
4. [BTS[4]](configs/BTS/README.md)

请点击上方的超链接查看每个算法的实现细节

## 安装

你可以通过如下命令下载PaddleMono工具箱

```
git clone https://github.com/PaddlePaddle/PaddleDepth.git
cd Paddle-Mono
pip install -r requirements.txt
```
请在Python 3.9中使用PaddleMono.

## 数据集准备
你可以参照 [dataset_prepare](data_prepare/data_prepare.md) 来进行数据集的准备.

## 如何使用


1. 在 `configs` 文件夹中修改相应模型的配置文件.
2. 运行 `train.py` 来训练config指定的模型, 例如: `CUDA_VISIBLE_DEVICES=0 python train.py --config configs/monodepthv2/mdp.yml`，若使用单机多卡，则运行`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch train.py --config configs/monodepthv2/mdp.yml --num_gpus 4`

* 我们提供shell脚本来帮助您复现我们的实验结果: `bash configs/monodepth/mdp.sh`.

## 定制化算法

你可以按照如下步骤开发自己的算法:

1. 检查你的模型是否需要模型来进行训练，如果有把模型加入到 `model`中
2. 加入你自己的配置文件（.sh 或 .yml）

## 结果

我们在KITTI数据集上根据通用的Eigen划分方法评测了PaddleMono已经实现的算法。

注意我们并没有通过额外的技巧来优化Monodepth2模型的结果，因此你可以直接使用.sh的脚本文件来复现我们在表格中报告的精度。

[comment]: <> (对于MLDA-Net，目前还没有完全对齐，表中给出torch权重转为paddle权重之后的测试精度。)

### KITTI

|     Method        | abs_rel | sq_rel | rms | log_rms | a1  | a2  | a3 |
|-------------|-------|-------|-------|-------|--------|--------|---------|
| Monodepth2_640x192 | 0.112 | 0.839 | 4.846 | 0.193 | 0.875  | 0.957 | 0.980   |
| Monodepth2_1024x32 | 0.112 | 0.833 | 4.748 | 0.191 | 0.880  | 0.960 | 0.981   |
| Depth Hints_640x192 | 0.110 | 0.818 | 4.728 | 0.189 | 0.881  | 0.959 | 0.981   |
| Depth Hints_1024x320 | 0.109 | 0.794 | 4.474 | 0.185 | 0.887  | 0.963 | 0.982   |
| MLDANet_640x192 | 0.108 | 0.829 | 4.678 | 0.184 | 0.885  | 0.962 | 0.983   |
| BTS Densenet121_704x352 | 0.050 | 0.201 | 2.547 | 0.082 | 0.970  | 0.995 | 0.999   |

## 贡献

PaddleMono工具箱目前还在积极维护与完善过程中。 我们非常欢迎外部开发者为Paddle Depth提供新功能\模型。 如果您有这方面的意愿的话，请往我们的邮箱或者issue里面反馈
## 感谢
PaddleDepth 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。
我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 
我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 参考文献

[1] Godard C, Mac Aodha O, Firman M, et al. Digging into self-supervised monocular depth estimation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 3828-3838.

[2] Song X, Li W, Zhou D, et al. MLDA-Net: Multi-level dual attention-based network for self-supervised monocular depth estimation[J]. IEEE Transactions on Image Processing, 2021, 30: 4691-4705.

[3] Watson J, Firman M, Brostow G J, et al. Self-supervised monocular depth hints[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 2162-2171.

[4] LEE J, HAN M, KO D, et al. From big to small: Multi-scale local planar guidance for monocular depth estimation[Z]//arXiv: Computer Vision and Pattern Recognition. 2019.

[comment]: <> "## Citation"

[comment]: <> "If you think this toolkit or the results are helpful to you and your research, please cite us!"

[comment]: <> "```"

[comment]: <> "@Misc{deepda,"

[comment]: <> "howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   "

[comment]: <> "title = {DeepDA: Deep Domain Adaptation Toolkit},  "

[comment]: <> "author = {Wang, Jindong and Hou, Wenxin}"

[comment]: <> "}  "

[comment]: <> "```"



## 联系方式

- [Yian Zhao](https://github.com/Zhao-Yian/): zhaoyian.zh@gmail.com
- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@pku.edu.cn
- [Bopei Zheng](https://github.com/zbp-xxxp/): bopei.zheng@foxmail.com
