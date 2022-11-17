# 准备 DSR-TestData 数据集

<div align="center">

[English](../../en_US/datasets/DSR-TestData.md)| [简体中文](DSR-TestData.md)

</div>

<!-- [DATASET] -->

```bibtex
@ARTICLE{9563214,
  author={Song, Xibin and Zhou, Dingfu and Li, Wei and Dai, Yuchao and Liu, Liu and Li, Hongdong and Yang, Ruigang and Zhang, Liangjun},
  journal={IEEE Transactions on Multimedia}, 
  title={WAFP-Net: Weighted Attention Fusion Based Progressive Residual Learning for Depth Map Super-Resolution}, 
  year={2022},
  volume={24},
  pages={4113-4127},
  doi={10.1109/TMM.2021.3118282}}
```

```text
├── test_data/
│   ├── cones_x4.mat
│   ├── teddy_x4.mat
│   ├── tskuba_x4.mat
│   └── venus_x4.mat
```

`DSR-TestData` 包括 `teddy`、 `cones`、 `tskuba` 和 `venus`四张深度图。 你能够从该 [网站](https://videotag.bj.bcebos.com/Data/WAFP_test_data.zip) 上下载该数据集。然后，你需要解压缩并移动相应的数据集，以遵循上述的文件夹结构。这些数据集都是由原作者精心准备的。