# BTS
A paddle implementation of the paper From big to small: Multi-scale local planar guidance for monocular depth estimation
[\[arXiv: Computer Vision and Pattern Recognition\]](https://arxiv.org/abs/1907.10326v5)


## Abstract
A supervised monocular depth estimation network. BTS propose a network architecture that utilizes novel local planar guidance layers located at multiple stages in the decoding phase.

## Training
The code for ***BTS*** builds upon [BTS](configs/BTS/README.md).

[comment]: <> (<!-- **KITTI Datasets Pretraining**)

[comment]: <> (Run the script `./configs/bts/bts.sh` to pre-train on KITTI datsets. Please update `--data_path` in the bash file as your training data path.)

[comment]: <> (**Finetuning**)

[comment]: <> (After training on 640x192 resolution, increase the resolution to 1024x320 for fine-tuning.)

[comment]: <> (Run the script `./configs/mldanet/mldanet.sh` to jointly finetune the pre-train model on KITTI dataset. )

[comment]: <> (Please update `--data_path` and `--load_weights_folder` as your training data path and pretrained weights folder. -->)

[comment]: <> (The training code of MLDANet has not been aligned yet. The [paddle weights]&#40;&#41; of MLDANet provided by PaddleMono are converted from the corresponding torch weights, and this part will be released later.)

## Evaluation
run the script `./configs/bts/bts.sh` to evaluate the model.

## Models
[backbone weights](https://pan.baidu.com/s/1uYSmKx04afm7e1ji0vA5qg?pwd=9fxg)
提取码：9fxg 

You can use this checkpoint to load the Model.

Please put backbone weights in the same directory and specify `load_weights_folder` 
as the directory path of pretraining model weights, i.e., `weights/weights_best_704x352/` when running the `bts.sh`.

```text
|-- weights/weights_best_704x352
  |-- encoder.pdparams
  |-- depth.pdparams
```

## Training
1. Modify the configuration file in the corresponding directories in `configs/BTS/bts.yml`. 
2. Run the `train_supervise.py` with specified config, for example, `python train_supervise.py --config configs/BTS/bts.yml`

## Citation
If you find this code useful in your research, please cite:
```
@article{lee2019big,
  title={From big to small: Multi-scale local planar guidance for monocular depth estimation},
  author={Lee, Jin Han and Han, Myung-Kyu and Ko, Dong Wook and Suh, Il Hong},
  journal={arXiv preprint arXiv:1907.10326},
  year={2019}
}
```
