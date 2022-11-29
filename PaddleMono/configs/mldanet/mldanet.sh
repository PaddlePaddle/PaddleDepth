#!/usr/bin/env bash
GPU_ID=0
data_dir=/data/kitti
depth_hints_path=/data/depth_hints
output_dir=output/mldanet/models/weights_best

# test the result of 192*640
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_depth.py --type MLDANet --load_weights_folder $output_fine_dir --eval_mono --data_path $test_dir --num_workers 4 --height 192 --width 640