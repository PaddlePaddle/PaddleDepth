# Paddle-DSR：A depth map super-resolution toolkit
</div>

<div align="center">

[English](README.md)| [简体中文](README_cn.md)

</div>


A lightweight, easy-to-extend, easy-to-learn, high-performance, and for-fair-comparison toolkit based on PaddlePaddle for Depth Super resolution. It is a part of the Paddledepth project

| cones| tskuba | teddy | venus |
| --- | --- | --- | ---|
| ![](https://ai-studio-static-online.cdn.bcebos.com/c16beee3e7c94284ae4e4b80f1f493af4477ef019b2a4efd9cb0c604b36be866)| ![](https://ai-studio-static-online.cdn.bcebos.com/9ccf5207aa1d4285b4f57c66bb5ae47b086c3df2d74d4c54b100b8d79e68f411)| ![](https://ai-studio-static-online.cdn.bcebos.com/ca98f5eb54ba4a0c8a275bd4afdd0c1ef45ac4e70d484762b0ad93745290d426)|![](https://ai-studio-static-online.cdn.bcebos.com/3137984e2b2342139e1dbaf78ab8abc49c869340f19743e7b804d632129cd413) |


## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

1. [WAFP-Net (IEEE Transactions on Multimedia 2021)[1]](docs/en_US/models/WAFP-Net.md)
2. [PMBANet (IEEE Transactions on Image Processing 2019)[2]](docs/en_US/models/PMBANet.md)
3. [RCAN (ECCV 2018)[3]](docs/en_US/models/RCAN.md)
4. [DRN (CVPR 2020)[4]](docs/en_US/models/DRN.md)

## Installation

You can install the Paddle-DSR toolbox by following steps:

- [Install PaddlePaddle ](https://www.paddlepaddle.org.cn/install/quick)
    - Version requirements: PaddlePaddle>=2.3.0, Python>=3.7

-  install Paddle-DSR toolbox

```
git clone https://github.com/PaddlePaddle/PaddleDepth.git
cd Depth_super_resolution
pip install -r requirements.txt
```
## Dataset 

see guidance in [dataset_prepare](docs/en_US/datasets) for [training](docs/zh_CN/datasets/data_all.md) and [testing](docs/zh_CN/datasets/DSR-TestData.md) dataset preparation.

## Usage

### Train

```shell
python -u tools/main.py --config-file $file_path$
```

- The `config-file` parameter is the path to the configuration file for the training model
- If there are pre-trained weights for finetune, run the following command to start training, with the `load` parameter being the path to the pre-trained weights

```shell
python -u tools/main.py --config-file $file_path$ --load $weight_path$
```

- If training is interrupted and needs to be resumed, run the following command, with the `resume` parameter as the checkpoint path

```shell
python -u tools/main.py --config-file $file_path$ --resume $checkpoint_path$
```

### Test


```shell
python -u tools/main.py --config-file $file_path$ --evaluate-only --load $weight_path$
```

## Customization

The file structure of Paddle-DSR is shown below:

```shell
Paddle-DSR
    │  README.md                
    │  README_cn.md             
    │  requirements.txt         
    ├─configs                   
    ├─data                      
    │  ├─process_DocumentIMG    
    │  ├─process_pmba           
    │  └─process_wafp           
    ├─docs                      
    ├─ppdsr 
    │  ├─datasets               
    │  │  └─preprocess          
    │  ├─engine                 
    │  ├─metrics                
    │  ├─models                 
    │  │  ├─backbones           
    │  │  ├─criterions          
    │  │  └─generators          
    │  ├─modules                
    │  ├─solver                 
    │  └─utils                  
    └─tools                     
```

You can develop your own algorithm by following these steps:

1. Check if your model needs a new loss function for training, and if so add the loss function to `ppdsr/models/criterions`.
2. Check if you need to add new models for training, if so add them to `ppdsr/models`.
3. check if you need to add new datasets for training, and if so add the datasets to `ppdsr/datasets`
4. Add your own configuration file (.yaml) to `configs`


## Results

We evaluated the algorithms already implemented in Paddle-DSR using the four depth maps `teddy`, `cones`, `tskuba` and `venus` as a test set `DSR-TestData`. 

**Note**: We did not optimize the model results by additional tricks, so you can directly use the .yaml configuration file to reproduce the accuracy we report in the table.

### DSR-TestData
|     Model        | PSNR | SSIM | RMSE | MAD | size  | 
|-------------|-------|-------|-------|-------|--------|
| WAFP-Net [1]| 42.0344 | 0.9834 | 2.5561 | 0.9246 | 3M | 
| PMBANet [2] | 41.0418 | 0.9825 | 2.4728 | 0.6278 | 94.9M  |
| RCAN [3]    | 42.5297 | 0.9850 | 2.4401 | 0.6685 | 59.6M  | 
| DRN [4]     | 42.4906 | 0.9850 | 2.4634 | 0.6506 | 18.4M  | 


## Contribution

The toolkit is under active development and contributions are welcome! 
Feel free to submit issues or emails to ask questions or contribute your code. 
If you would like to implement new features, please submit a issue or emails to discuss with us first.

## References

[1] Song, Xibin, Dingfu Zhou, Wei Li, Yuchao Dai, Liu Liu, Hongdong Li, Ruigang Yang, and Liangjun Zhang. ‘WAFP-Net: Weighted Attention Fusion Based Progressive Residual Learning for Depth Map Super-Resolution’. IEEE Transactions on Multimedia 24 (2022): 4113–27. https://doi.org/10.1109/TMM.2021.3118282.
.

[2] Ye, Xinchen, Baoli Sun, Zhihui Wang, Jingyu Yang, Rui Xu, Haojie Li, and Baopu Li. ‘PMBANet: Progressive Multi-Branch Aggregation Network for Scene Depth Super-Resolution’. IEEE Transactions on Image Processing 29 (2020): 7427–42. https://doi.org/10.1109/TIP.2020.3002664.

[3] Zhang, Yulun, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. ‘Image Super-Resolution Using Very Deep Residual Channel Attention Networks’. In Computer Vision – ECCV 2018, edited by Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, 11211:294–310. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2018. https://doi.org/10.1007/978-3-030-01234-2_18.


[4] Guo, Yong, Jian Chen, Jingdong Wang, Qi Chen, Jiezhang Cao, Zeshuai Deng, Yanwu Xu, and Mingkui Tan. ‘Closed-Loop Matters: Dual Regression Networks for Single Image Super-Resolution’. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5406–15. Seattle, WA, USA: IEEE, 2020. https://doi.org/10.1109/CVPR42600.2020.00545.


## Contact

- [Yuanhang Kong](https://github.com/kongdebug): 2111330@tongji.edu.cn
- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@baidu.com
