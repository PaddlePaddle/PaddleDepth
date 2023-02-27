# Monodepthv2(CVPR 2019)
A paddle implementation of the paper Digging Into Self-Supervised Monocular Depth Estimation
[\[ICCV2019\]](https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html)


## Abstract
Per-pixel ground-truth depth data is challenging to acquire at scale. To overcome this limitation, self-supervised learning has emerged as a promising alternative for training models to perform monocular depth estimation. In this paper, we propose a set of improvements, which together result in both quantitatively and qualitatively improved depth maps compared to competing self-supervised methods. Research on self-supervised monocular training usually explores increasingly complex architectures, loss functions, and image formation models, all of which have recently helped to close the gap with fully-supervised methods. We show that a surprisingly simple model, and associated design choices, lead to superior predictions. In particular, we propose (i) a minimum reprojection loss, designed to robustly handle occlusions, (ii) a full-resolution multi-scale sampling method that reduces visual artifacts, and (iii) an auto-masking loss to ignore training pixels that violate camera motion assumptions. We demonstrate the effectiveness of each component in isolation, and show high quality, state-of-the-art results on the KITTI benchmark.


## Training
**KITTI Datasets Pretraining**

Run the script `./configs/monodepthv2/mdp.sh` to pre-train on KITTI datsets. Please update `--data_path` in the bash file as your training data path and specify `weights_init` as the directory path of backbone weights, i.e., `/root/paddlejob/shenzhelun/PaddleMono-master/weights/backbone_weight/resnet18-pytorch`.

**Finetuning**

After training on 640x192 resolution, increase the resolution to 1024x320 for fine-tuning.
Run the script `./configs/monodepthv2/mdp.sh` to jointly finetune the pre-train model on KITTI dataset. 
Please update `--data_path` and `--load_weights_folder` as your training data path and pretrained weights folder.

## Evaluation

run the script `./configs/monodepthv2/mdp.sh` to evaluate the model.

## Models

[Pretraining Model](https://drive.google.com/file/d/14hUDOt4lt6glPdUAwgky523E1yZJ1Wvw/view?usp=sharing)

You can use this checkpoint to reproduce the result of Monodepth2_640x192.

[comment]: <> (You can use this checkpoint to reproduce the result of Monodepth2_640x192.)

[Finetuning Model](https://drive.google.com/file/d/1OQDwQ0MI-XA2GDBsCr0E4CSc0BJPqnCB/view?usp=sharing)

You can use this checkpoint to reproduce the result of Monodepth2_1024x320. 

[comment]: <> (You can use this checkpoint to reproduce the result of Monodepth2_1024x320.)

[backbone weights](https://drive.google.com/file/d/1iVnt_6I0u2U4wo1ZeG1Iy2DvZ1Ltn-2l/view?usp=share_link)

You can use this checkpoint to load the backbone weights of resnet18.

Please put pretraining model weights and backbone weights in the same directory and specify `load_weights_folder` 
as the directory path of pretraining model weights, i.e., `weights/weights_best_640x192/` when running the `mdp.sh`.

```text
|-- weights/weights_best_640x192
  |-- resnet18_pretrain.h5
  |-- encoder.pdparams
  |-- depth.pdparams
  |-- pose_encoder.pdparams
  |-- pose.pdparams
```

If you want to put the backbone weights on the other directory, please further specify `weights_init` as the directory path of backbone weights, i.e., `/root/paddlejob/shenzhelun/PaddleMono-master/weights/backbone_weight/resnet18-pytorch`

## Citation
If you find this code useful in your research, please cite:
```
@inproceedings{godard2019digging,
  title={Digging into self-supervised monocular depth estimation},
  author={Godard, Cl{\'e}ment and Mac Aodha, Oisin and Firman, Michael and Brostow, Gabriel J},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3828--3838},
  year={2019}
}
```