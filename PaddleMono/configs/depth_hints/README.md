# Self-Supervised Monocular Depth Hints(ICCV 2019)
A paddle implementation of the paper Self-Supervised Monocular Depth Hints.
[\[ICCV 2019\]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Watson_Self-Supervised_Monocular_Depth_Hints_ICCV_2019_paper.pdf)


## Abstract
Monocular depth estimators can be trained with various forms of self-supervision from binocular-stereo data to circumvent the need for high-quality laser-scans or other ground-truth data. The disadvantage, however, is that the photometric reprojection losses used with self-supervised learning typically have multiple local minima. These plausible-looking alternatives to ground-truth can restrict what a regression network learns, causing it to predict depth maps of limited quality. As one prominent example, depth discontinuities around thin structures are often incorrectly estimated by current state-of-the-art methods. Here, we study the problem of ambiguous reprojections in depth-prediction from stereo-based self-supervision, and introduce Depth Hints to alleviate their effects. Depth Hints are complementary depth-suggestions obtained from simple off-the-shelf stereo algorithms. These hints enhance an existing photometric loss function, and are used to guide a network to learn better weights. They require no additional data, and are assumed to be right only sometimes. We show that using our Depth Hints gives a substantial boost when training several leading self-supervised-from-stereo models, not just our own. Further, combined with other good practices, we produce state-of-the-art depth predictions on the KITTI benchmark.

## Training

The code for ***Depth Hints*** builds upon [Monodepth2](configs/monodepthv2/README.md).

To train using depth hints:
  - Clone this repository
  - Run `python precompute_depth_hints.py  --data_path <your_KITTI_path>`, optionally setting `--save_path` (will default to <data_path>/depth_hints) and `--filenames` (will default to training and validation images for the eigen split). This will create the "fused" depth hints referenced in the paper. This process takes approximately 4 hours on a GPU.
  At present, there are still some problems in calculating the recurrence of depth hints. Currently, the depth hint dataset generation is not supported for paddledepth. You can use the torch version of the script to generate the depth hints of dataset.
  If you want to use it, please click [here](https://github.com/nianticlabs/depth-hints/blob/master/precompute_depth_hints.py).
  - Add the flag `--use_depth_hints` to your usual monodepth2 training command, optionally also setting `--depth_hint_path` (will default to <data_path>/depth_hints). See below for a full command.
  
**KITTI Datasets Pretraining**

Run the script `./configs/depth_hints/depth_hints.sh` to pre-train on KITTI datsets. Please update `--data_path` in the bash file as your training data path and specify `weights_init` as the directory path of backbone weights, i.e., `/root/paddlejob/shenzhelun/PaddleMono-master/weights/backbone_weight/resnet18-pytorch`.

**Finetuning**

After training on 640x192 resolution, increase the resolution to 1024x320 for fine-tuning.
Run the script `./configs/depth_hints/depth_hints.sh` to jointly finetune the pre-train model on KITTI dataset. 
Please update `--data_path` and `--load_weights_folder` as your training data path and pretrained weights folder.

## Evaluation

run the script `./configs/depth_hints/depth_hints.sh` to evaluate the model.

## Models

[Pretraining Model](https://drive.google.com/file/d/1z3_ehxeDdmaQwSlUBXblF-mDnWl48hbK/view?usp=share_link)

You can use this checkpoint to reproduce the result of depth_hints_640x192.

[Finetuneing Model](https://drive.google.com/file/d/198qXsrIV2d6K5layPTFPeXI3wmm_aFWx/view?usp=share_link)

You can use this checkpoint to reproduce the result of depth_hints_1024x320.

[backbone weights](https://drive.google.com/file/d/1iVnt_6I0u2U4wo1ZeG1Iy2DvZ1Ltn-2l/view?usp=share_link)

You can use this checkpoint to load the backbone weights of resnet18.

[comment]: <> (Please put pretraining model weights and backbone weights in the same directory &#40;or different&#41;, and then specify `weights_init` as the directory path of backbone weights and specify `load_weights_folder` as the directory path of pretraining model weights when running the `evaluate_depth.py`.)
Please put pretraining model weights and backbone weights in the same directory and specify `load_weights_folder` 
as the directory path of pretraining model weights, i.e., `weights/weights_best_640x192/` when running the `depth_hints.sh`.

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
@inproceedings{watson-2019-depth-hints,
  title     = {Self-Supervised Monocular Depth Hints},
  author    = {Jamie Watson and
               Michael Firman and
               Gabriel J. Brostow and
               Daniyar Turmukhambetov},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```