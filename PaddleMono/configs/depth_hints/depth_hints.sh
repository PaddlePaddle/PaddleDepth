#!/usr/bin/env bash
GPU_ID=0
data_dir=/data/kitti
test_dir=/data/eigen
depth_hints_path=/data/depth_hints
log_dir=output/
log_fine_dir=output_fine/
output_dir=output/depth_hints/models/weights_best
output_fine_dir=output_fine/depth_hints/models/weights_best

# train in KITTI with depth hints
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --config configs/monodepthv2/mdp.yml --log_dir $log_dir --data_dir $data_dir --use_stereo --use_depth_hints --depth_hint_path $depth_hints_path

# fintune in KITTI
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --config configs/monodepthv2/mdp.yml --data_path $data_dir --height 320 --width 1024 --use_stereo --log_dir $log_fine_dir --load_weights_folder $output_dir --batch_size 4 --num_epochs 2 --learning_rate 5e-5

# test the result of 192*640
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_depth.py --type MonoDepthv2 --load_weights_folder $output_fine_dir --eval_mono --data_path $test_dir --num_workers 4 --height 192 --width 640

# test the result of 320*1024
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_depth.py --type MonoDepthv2 --load_weights_folder $output_fine_dir --eval_mono --data_path $test_dir --num_workers 4 --height 320 --width 1024
