# PMBANet(IEEE Transactions on Image Processing 2020)

<div align="center">

[English](PMBANet.md)| [简体中文](../../zh_CN/models/PMBANet.md)

</div>

A paddle implementation of the paper PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution,
[\[IEEE Transactions on Image Processing 2020\]](https://ieeexplore.ieee.org/document/9127098)

## Abstract
Depth map super-resolution is an ill-posed inverse problem with many challenges. First, depth boundaries are generally hard to reconstruct particularly at large magnification factors. Second, depth regions on fine structures and tiny objects in the scene are destroyed seriously by downsampling degradation. To tackle these difficulties, we propose a progressive multi-branch aggregation network (PMBANet), which consists of stacked MBA blocks to fully address the above problems and progressively recover the degraded depth map. Specifically, each MBA block has multiple parallel branches: 1) The reconstruction branch is proposed based on the designed attention-based error feed-forward/-back modules, which iteratively exploits and compensates the downsampling errors to refine the depth map by imposing the attention mechanism on the module to gradually highlight the informative features at depth boundaries. 2) We formulate a separate guidance branch as prior knowledge to help to recover the depth details, in which the multi-scale branch is to learn a multi-scale representation that pays close attention at objects of different scales, while the color branch regularizes the depth map by using auxiliary color information. Then, a fusion block is introduced to adaptively fuse and select the discriminative features from all the branches. The design methodology of our whole network is well-founded, and extensive experiments on benchmark datasets demonstrate that our method achieves superior performance in comparison with the state-of-the-art methods.


## Data prepare

The operations for downloading and producing high-resolution and low-resolution depth image pairs are the same as for [DRN](docs/en_US/models/DRN.md). For PMBANet, as the required image blocks are relatively small, the `data/process_pmba/process_pmba_data.py` script is executed to slice the resulting depth map image pairs according to `crop_size = 128` with `step = 64`

**Note**: The processed dataset has been uploaded to the AI Studio platform at the following link: https://aistudio.baidu.com/aistudio/datasetdetail/173618

## Training

The following command is executed to train the PMBANet using the `PMBA` dataset

```shell
python -u tools/main.py --config-file configs/pmba_x4.yaml
```


## Evaluation
**DSR-TestData**

Execute the following command to test the `DSR-TestData` dataset
```shell
python -u tools/main.py --config-file configs/pmba_x4.yaml --evaluate-only --load pmba_x4_best.pdparams
```


## Models

[Pretraining Model](https://aistudio.baidu.com/aistudio/datasetdetail/176907)
You can use this trained weight to reproduce the results reported in [README.md](README.md)

## Citation
If you find this code useful in your research, please cite:
```
@ARTICLE{9127098,
  author={Ye, Xinchen and Sun, Baoli and Wang, Zhihui and Yang, Jingyu and Xu, Rui and Li, Haojie and Li, Baopu},
  journal={IEEE Transactions on Image Processing}, 
  title={PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution}, 
  year={2020},
  volume={29},
  pages={7427-7442},
  doi={10.1109/TIP.2020.3002664}}
```