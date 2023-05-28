#!/usr/bin/env bash
GPU_ID=0
output_dir=output/bts/models/weights_best

# test the result of 352*704
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --type BTS --load_weights_folder /home/aistudio/work/weights/weights_best_704x352 --encoder "densenet121_bts" --eval_mono --dataset "kitti_supervise" --data_path "/home/aistudio/data/data192117/KITTI_bts_test" --num_workers 4 --height 352 --width 704 --png