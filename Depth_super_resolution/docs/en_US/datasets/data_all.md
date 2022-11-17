# Prepare data_all dataset

<div align="center">

[English](data_all.md)| [简体中文](../../zh_CN/datasets/data_all.md)

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
├── data_all/
│   ├── alley_1_1.png
│   ├── ...
│   └── ...
```

The `data_all` dataset combines the [Middlebury dataset](https://vision.middlebury.edu/stereo/data/) / [MPI Sintel dataset](http://sintel.is.tue.mpg.de/downloads) and the [synthetic New Tsukuba dataset](https://en.home.cvlab.cs.tsukuba.ac.jp/dataset).

You can download datasets on this [webpage](https://videotag.bj.bcebos.com/Data/WAFP_data.zip). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.