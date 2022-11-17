# Prepare DSR-TestData dataset

<div align="center">

[English](DSR-TestData.md)| [简体中文](../../zh_CN/datasets/DSR-TestData.md)

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

The `DSR-TestData` dataset combines four depth maps `teddy`, `cones`, `tskuba` and `venus`. You can download datasets on this [webpage](https://videotag.bj.bcebos.com/Data/WAFP_test_data.zip). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.