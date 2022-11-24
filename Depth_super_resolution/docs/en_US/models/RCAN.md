# RCAN(ECCV 2018)

<div align="center">

[English](RCAN.md)| [简体中文](../../zh_CN/models/RCAN.md)

</div>

A paddle implementation of the paper RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks,
[\[ECCV 2018\]](https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html)

## Abstract

Convolutional neural network (CNN) depth is of crucial importance for image super-resolution (SR). However, we observe that deeper networks for image SR are more difficult to train. The low-resolution (LR) inputs and features contain abundant low-frequency information, which is treated equally across channels, hence hindering the representational ability of CNNs. To solve these problems, we propose the very deep residual channel attention networks (RCAN). Specifically, we propose residual in residual (RIR) structure to form very deep network, which consists of several residual groups with long skip connections. Each residual group contains some residual blocks with short skip connections. Meanwhile, RIR allows abundant low-frequency information to be bypassed through multiple skip connections, making the main network focus on learning high-frequency information. Furthermore, we propose channel attention mechanism to adaptively rescale channel-wise features by considering interdependencies among channels. Extensive experiments show that our RCAN achieves better accuracy and visual improvements against state-of-the-art methods.

## Data prepare

The same as [DRN Data prepare](docs/en_US/models/DRN.md).

## Training

According to the paper, RCAN trains super-resolution 4x models that require super-resolution 2x weights as pre-training weights. Execute the following command to download the super-resolution 2x pre-training weights trained on the `DIV2K` dataset

```shell
wget https://paddlegan.bj.bcebos.com/models/RCAN_X2_DIV2K.pdparams
```

As the RCAN network is designed for the image super-segmentation task, the input channels to the network are reserved for 3. The following command is executed to train the RCAN using the `PMBA` dataset

```shell
python -u tools/main.py --config-file configs/rcan_dsr_x4.yaml --load ./RCAN_X2_DIV2K.pdparams
```

## Evaluation
**DSR-TestData**

Execute the following command to test the `DSR-TestData` dataset
```shell
python -u tools/main.py --config-file configs/rcan_dsr_x4.yaml --evaluate-only --load rcan_x4_best.pdparams
```

## Models

[Pretraining Model](https://aistudio.baidu.com/aistudio/datasetdetail/176907)
You can use this trained weight to reproduce the results reported in [README.md](README.md)


## Citation
If you find this code useful in your research, please cite:
```
@InProceedings{Zhang_2018_ECCV,
author = {Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
title = {Image Super-Resolution Using Very Deep Residual Channel Attention Networks},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
} 
```