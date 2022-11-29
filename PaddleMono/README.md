# PaddleMono: A Unified Framework for Monocular Depth Estimation
</div>

<div align="center">

[English](README.md)| [简体中文](README_zh-CN.md)

</div>
A lightweight, easy-to-extend, easy-to-learn, high-performance, and for-fair-comparison toolkit based 
on PaddlePaddle for Monocular Depth Estimation. It is a part of the Paddledepth project.


## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.


[comment]: <> (- Monodepth2)

[comment]: <> (- MLDA-Net)

[comment]: <> (- Depth Hints &#40;This can be used in the training of the above two models&#41;)


1. [Monodepth2 (ICCV2019)[1]](configs/monodepthv2/README.md)
2. [MLDA-Net (TIP2021)[2]](configs/mldanet/README.md)
3. [Depth Hints (ICCV2019)[3]](configs/depth_hints/README.md)

Please click the hyperlink of each algorithm for more detailed explanation.

## Installation
You can either git clone this whole repo by:
```
git clone https://github.com/PaddlePaddle/PaddleDepth.git
cd Paddle-Mono
pip install -r requirements.txt
```
Please use PaddleMono in python 3.9.

## Dataset 
see guidance in [dataset_prepare](data_prepare/data_prepare.md) for dataset preparation.

## Usage

1. Modify the configuration file in the corresponding directories in `configs`. 
2. Run the `train.py` with specified config, for example, `python train.py  --config configs/monodepthv2/mdp.yml`

* We provide shell scripts to help you reproduce our experimental results: `bash configs/monodepth/mdp.sh`.

## Customization

It is easy to design your own method with following steps:

1. Check and write your own model's to `model`
2. Write your own config file (.sh or .yml)

## Results

We present results of our implementations on the popular KITTI benchmarks with Eigen split. 

**Note:** We did not optimize the model results by additional tricks, so you can directly use the provided shell scripts to reproduce the accuracy we report in the table.

[comment]: <> (We did not perform careful parameter tuning and simply used the default config files for the . )

[comment]: <> (You can easily reproduce our results using provided shell scripts!)

[comment]: <> (For MLDA Net, it is not fully aligned yet. )

[comment]: <> (We show the test accuracy using the torch weights converted to paddle.)

### KITTI

|     Method        | abs_rel | sq_rel | rms | log_rms | a1  | a2  | a3 |
|-------------|-------|-------|-------|-------|--------|--------|---------|
| Monodepth2_640x192 | 0.112 | 0.839 | 4.846 | 0.193 | 0.875  | 0.957 | 0.980   |
| Depth Hints_640x192 | 0.110 | 0.818 | 4.728 | 0.189 | 0.881  | 0.959 | 0.981   |
| Depth Hints_1024x320 | 0.109 | 0.794 | 4.474 | 0.185 | 0.887  | 0.963 | 0.982   |
| MLDANet_640x192 | 0.108 | 0.829 | 4.678 | 0.184 | 0.885  | 0.962 | 0.983   |


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

[1] Godard C, Mac Aodha O, Firman M, et al. Digging into self-supervised monocular depth estimation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 3828-3838.

[2] Song X, Li W, Zhou D, et al. MLDA-Net: Multi-level dual attention-based network for self-supervised monocular depth estimation[J]. IEEE Transactions on Image Processing, 2021, 30: 4691-4705.

[3] Watson J, Firman M, Brostow G J, et al. Self-supervised monocular depth hints[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 2162-2171.

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

- [Yian Zhao](https://github.com/Zhao-Yian/): zhaoyian.zh@gmail.com
- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@pku.edu.cn