#!/bin/bash
gou_id=1,2,3,4
CUDA_VISIBLE_DEVICES=${gou_id} nohup python3 -u -m paddle.distributed.launch --gpus=${gou_id} Source/main.py \
                        --mode train \
                        --batchSize 1 \
                        --gpu 4 \
                        --trainListPath ./Datasets/Stereo/kitti2012_training_list.csv \
                        --imgWidth 448 \
                        --imgHeight 256 \
                        --dataloaderNum 4 \
                        --maxEpochs 300 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --auto_save_num 50 \
                        --dist False \
                        --modelName PWCNet \
                        --outputDir ./DebugResult/ \
                        --modelDir ./DebugCheckpoint/ \
                        --dataset kitti2012 > TrainRun.log 2>&1 &

tail -f  TrainRun.log
