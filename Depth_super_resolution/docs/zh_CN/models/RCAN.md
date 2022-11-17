# RCAN(ECCV 2018)

<div align="center">

[English](../../en_US/models/RCAN.md)| [简体中文](DRN.md)

</div>

A paddle implementation of the paper RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks,
[\[ECCV 2018\]](https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html)


## 摘要

卷积神经网络（CNN）的深度对于图像超级分辨率（SR）来说是至关重要的。然而，我们观察到，用于图像SR的更深的网络更难训练。低分辨率（LR）输入和特征包含丰富的低频信息，这些信息在不同的通道上被平等对待，因此阻碍了CNN的表示能力。为了解决这些问题，我们提出了非常深入的残差通道注意网络（RCAN）。具体来说，我们提出了残差（RIR）结构来形成非常深的网络，它由几个具有长跳接的残差组组成。每个残差组包含一些具有短跳过连接的残差块。同时，RIR允许丰富的低频信息通过多个跳过连接被绕过，使主网络专注于学习高频信息。此外，我们提出了信道关注机制，通过考虑信道之间的相互依存关系，自适应地重新划分信道的特征。大量的实验表明，与最先进的方法相比，我们的RCAN实现了更好的准确性和视觉上的改进


## 数据准备

与[DRN数据准备](docs/zh_CN/models/DRN.md)一致

## 训练

由于RCAN网络是为图像超分任务设计，保留网络的输入通道为3。执行以下命令，使用`PMBA`数据集训练RCAN

```shell
python -u tools/main.py --config-file configs/rcan_dsr_x4.yaml
```

## 测试

**DSR-TestData**
执行以下命令，对`DSR-TestData`数据集进行测试
```shell
python -u tools/main.py --config-file configs/rcan_dsr_x4.yaml --evaluate-only --load rcan_x4_best.pdparams
```

## 模型

[Pretraining Model](https://aistudio.baidu.com/aistudio/datasetdetail/176907)
你可以使用这个训练好的权重来重现[README_cn.md](README_cn.md)中报告的结果


## 引用

如果你觉得代码对你的研究有帮助，请引用
```
@InProceedings{Zhang_2018_ECCV,
author = {Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
title = {Image Super-Resolution Using Very Deep Residual Channel Attention Networks},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
} 
```