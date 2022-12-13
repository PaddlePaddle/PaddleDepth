#!/bin/bash
gou_id=0
CUDA_VISIBLE_DEVICES=${gou_id} python3 Source/main.py \
                        --mode train \
                        --batchSize 2 \
                        --gpu 1 \
                        --trainListPath ./Datasets/Stereo/rob_training_list_aistudio.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 0 \
                        --maxEpochs 300 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --auto_save_num 10 \
                        --dist False \
                        --modelName PSMNet \
                        --outputDir ./DebugResult/ \
                        --modelDir ./SceneFLow/model_epoch_20.pth \
                        --dataset kitti2012
