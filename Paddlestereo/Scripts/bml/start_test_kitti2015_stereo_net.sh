#!/bin/bash
# parameter
test_gpus_id=0
eva_gpus_id=0
test_list_path='./Datasets/Stereo/kitti2015_testing_list_aistudio.csv'
evalution_format='training'

rm -r ResultImg/
echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "start to predict disparity map"
CUDA_VISIBLE_DEVICES=${test_gpus_id} python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ${test_list_path} \
                        --imgWidth 1536 \
                        --imgHeight 512 \
                        --dataloaderNum 0 \
                        --maxEpochs 45 \
                        --imgNum 200 \
                        --log ./TestLog/ \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --port 6123 \
                        --dist False \
                        --modelName PSMNet \
                        --outputDir ./DebugResult/ \
                        --modelDir ./KITTI2015/model_epoch_99.pth \
                        --dataset kitti2015

if grep -q ${evalution_format} ${test_list_path}; then
    echo "start to evalulate disparity map"
    echo "evalution gpus id: "${eva_gpus_id}
    CUDA_VISIBLE_DEVICES=${eva_gpus_id} python ./Source/Tools/evalution_stereo_net.py --gt_list_path ${test_list_path}
fi

echo "Begin packaging the result"
cp -r ResultImg/ disp_0/
zip -r disp_0.zip disp_0/
rm -r disp_0/
mv disp_0.zip ResultImg
echo "Finish!"