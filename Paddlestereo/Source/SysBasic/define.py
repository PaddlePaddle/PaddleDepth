# -*- coding: utf-8 -*-
# default program's setting
# the path's define, you can change the content
# output's path
DATA_OUTPUT_PATH = './Result/'
# model's path
MODEL_PATH = './Checkpoint/'
# Training list, we use it training and testing
TRAIN_LIST_PATH = './Datasets/msd_training_list.csv'
# Validation list
VAL_LIST_PATH = './Datasets/msd_val_list.csv'
# result image's path
RESULT_OUTPUT_PATH = './ResultImg/'
# log path
LOG_OUTPUT_PATH = './log/'
# The user's directory
USER_DIRECTORY = './Source/UserModelImplementation/'

# default program's name
# Dataset's name
DATASET_NAME = 'Default_Dataset'
# Model's name
MODEL_NAME = 'Default_Model'
# Checkpoint's list file
CHECK_POINT_LIST_NAME = 'checkpoint.list'
# Checkpoint's name
# model_epoch.pth
CHECK_POINT_NAME = 'model_epoch_%d.pdparams'
#
LAST_MODEL_NAME = 'last model name:'

# define name pipe
PIPE_WRITE_PATH = '/tmp/jf_pipe.in'
PIPE_READ_PATH = '/tmp/jf_pipe.out'

# image's setting
# We use this data to crop the image in training process;
# We ues this data to pad the image in testing process.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# the sample number
SAMPLE_NUM = 128
# the number of test
TEST_NUM = 1
# MAX_EPOCHS
MAX_EPOCHS = 100
# IP
IP = 'localhost'
# Port
PORT = '8886'
# dist
DIST = True

# the learning settings
# batch size
BATCH_SIZE = 64
# learn rate
LEARNING_RATE = 0.001
# save path
AUTO_SAVE_NUM = 1
# The number of GPU
GPU_NUM = 2
# the number of Dataloader
DATA_LOADER_NUM = 4
# the training and val's image number
IMG_NUM = 4500
VAL_IMG_NUM = 0

VERSION = '0.1.0'
