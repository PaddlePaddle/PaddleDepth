#!/bin/bash
gpu_id=0,1,2,3
CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 -u -m paddle.distributed.launch --gpus=${gpu_id} Source/main.py \
                        --mode train \
                        --batchSize 2 \
                        --gpu 4 \
                        --trainListPath ./Datasets/Stereo/kitti2015_training_list.csv \
                        --imgWidth 1000 \
                        --imgHeight 320 \
                        --dataloaderNum 4 \
                        --maxEpochs 200 \
                        --imgNum 200 \
                        --sampleNum 1 \
                        --lr 0.0002 \
                        --auto_save_num 50 \
                        --dist False \
                        --modelName RAFT_STEREO \
                        --precision fp16 \
                        --outputDir /root/paddlejob/workspace/output/ \
                        --modelDir /root/paddlejob/workspace/Paddlestereo/raft_sceneflow.pdparams \
                        --pretrain True \
                        --iter_update True \
                        --dataset kitti2015 > TrainRun.log 2>&1 &

tail -f  TrainRun.log