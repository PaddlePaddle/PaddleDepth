#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python -u Source/main.py \
                        --mode train \
                        --batchSize 1 \
                        --gpu 4 \
                        --trainListPath ./Datasets/Stereo/scene_flow_debug_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 0 \
                        --maxEpochs 45 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist False \
                        --modelName PSMNet \
                        --outputDir ./DebugResult/ \
                        --modelDir ./DebugCheckpoint/ \
                        --dataset sceneflow
