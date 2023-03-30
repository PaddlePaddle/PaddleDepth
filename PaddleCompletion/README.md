# PaddleCompletion: A Unified Framework for Depth Completion

<div align="center">

[English](README.md)| [简体中文](README_zh-CN.md)

</div>
A lightweight, easy-to-extend, easy-to-learn, high-performance, and for-fair-comparison toolkit based 
on PaddlePaddle for Depth Completion. It is a part of the Paddledepth project.


## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

1. [CSPN (ECCV2018)](model_document/CSPN/README.md)
2. [FCFRNet (AAAI2021)](model_document/FCFRNet/README.md)
3. [STD (ICRA2019)](model_document/STD/README.md)
4. 
Please click the hyperlink of each algorithm for more detailed explanation.

## Installation

You can git clone this whole repo by:

```
git clone https://github.com/PaddlePaddle/PaddleDepth
cd PaddleDepth/PaddleCompletion
pip install -r requirements.txt
```

This project is based on PaddlePaddle 2.3.2. Please use PaddleCompletion in python 3.9. 

## Dataset

See guidance in [dataset_prepare](data_prepare/data_prepare.md) for dataset preparation.

## Usage

### Training

1. Modify the .yaml file in the `configs` directory.
2. Run the `train.py` with specified config, eg: `python train.py --c config/CSPN.yaml`

* We provide shell scripts to help you reproduce our experimental results: `bash scripts/train_cspn.sh`.

### Evaluation

1. Modify the configuration file in the `configs` directory.
2. Download the trained model and put it in the corresponding directory, eg: `weights/cspn/model_best.pdparams`.
3. Run the `evaluate.py` with specified config, eg: `python evaluate.py --c config/CSPN.yaml`

## Customization

It is easy to design your own method following the 3 steps:

1. Check whether your method requires new loss functions, if so, add your loss in the `loss_funcs`
2. Check and write your own model's to `model`
3. Write your own config file (.yaml)

## Results

We present results of our implementations on 2 popular benchmarks: KITTI and NYU Depth V2. 
We did not perform careful parameter tuning and simply used the default config files. 
You can easily reproduce our results using provided shell scripts!


### KITTI

| Method    | RMSE    | MAE     | iRMSE | iMAE  |
|-----------| ------- | ------- | ----- | ----- |
| `FCFRNet` | 784.224 | 222.639 | 2.370 | 1.014 |
| `STD` | 814.73 | 242.639 | 2.80 | 1.21 |

### NYU Depth V2

| Data    | RMSE   | REL    | DELTA1.02 | DELTA1.05 | DELTA1.10 |
|---------| ------ | ------ | --------- | --------- | --------- |
| `CSPN`  | 0.1111 | 0.0151 | 0.8416    | 0.9386    | 0.9729    |


## Contribution

The toolkit is under active development and contributions are welcome! 
Feel free to submit issues or emails to ask questions or contribute your code. 
If you would like to implement new features, please submit a issue or emails to discuss with us first.

## Acknowledgement

PaddleDepth is an open source project that is contributed by researchers and engineers 
from various colleges and companies. 
We appreciate all the contributors who implement their methods or add new features, 
as well as users who give valuable feedbacks. 
We wish that the toolbox and benchmark could serve the growing research community by 
providing a flexible toolkit to reimplement existing methods and develop their new algorithms.

## References
    
[1] [CSPN: A Compact Spatial Propagation Network for Depth Completion](https://openaccess.thecvf.com/content_ECCV_2018/html/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.html)

[2] [FCFRNet: Fast and Convergent Feature Refinement Network for Depth Completion](https://doi.org/10.1609/aaai.v35i3.16311)

[comment]: <> (## Citation)

[comment]: <> (If you think this toolkit or the results are helpful to you and your research, please cite us!)

[comment]: <> (```)

[comment]: <> (@Misc{deepda,)

[comment]: <> (howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   )

[comment]: <> (title = {DeepDA: Deep Domain Adaptation Toolkit},  )

[comment]: <> (author = {Wang, Jindong and Hou, Wenxin})

[comment]: <> (}  )

[comment]: <> (```)



## Contact

- [Juntao Lu](https://github.com/ralph0813): juntao.lu@student.unimelb.edu.au
- [Bing Xiong](https://github.com/imexb9584): bingxiong9527@siat.edu.cn
- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@pku.edu.cn
