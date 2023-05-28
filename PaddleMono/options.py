# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="options")
        # CFG
        self.parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
        self.parser.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory")
                                 

        # TRAINING options
        self.parser.add_argument('--num_gpus',
                                 type=int,
                                 help='number of gpus used in training')
        self.parser.add_argument("--seed",
                                 type=int,
                                 help='seed used in training.')
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_lite", "eigen_full"])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--freeze_bn",
                                 action='store_true',
                                 help='freeze the running mean and running variance of all bn layers.')
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height")
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width")
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight")

        # The below three args please config in the yaml file.
        # self.parser.add_argument("--scales",
        #                          nargs="+",
        #                          type=int,
        #                          help="scales used in the loss",
        #                          default=[0, 1, 2, 3])
        # self.parser.add_argument("--min_depth",
        #                          type=float,
        #                          help="minimum depth",
        #                          default=0.1)
        # self.parser.add_argument("--max_depth",
        #                          type=float,
        #                          help="maximum depth",
        #                          default=100.0)
        # self.parser.add_argument("--frame_ids",
        #                          nargs="+",
        #                          type=int,
        #                          help="frames to load",
        #                          default=[0, -1, 1])
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")


        # DEPTH HINT options
        self.parser.add_argument("--use_depth_hints",
                                 help="if set, apply depth hints during training",
                                 action="store_true")
        self.parser.add_argument("--depth_hint_path",
                                 type=str,
                                 help="path to load precomputed depth hints from. If not set will.be    assumed to be data_path/depth_hints")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate")
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="number of epochs")
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs")
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler")
        self.parser.add_argument("--epsilon",
                                 type=float,
                                 help="epsilon in Adam optimizer",
                                 default=0.001)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay factor for optimization",
                                 default=0.01)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepthv2 v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="choose from default (paddle pretrained weights), scratch, or a path to a custom weight file.")
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers")

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each console log")
        self.parser.add_argument("--visualdl_frequency",
                                 type=int,
                                 help="number of batches between each visualdl log")
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save")

        # EVALUATION options
        # Those below args are not support, please config them in the yaml file if you need.
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepthv2 paper",
                                 action="store_true")

        # SUPERVISE options
        self.parser.add_argument("--encoder",
                                 type=str,
                                 help='type of encoder',
                                 default='densenet121_bts')
        self.parser.add_argument("--max_depth",
                                 type=float, 
                                 help='maximum depth in estimation', 
                                 default=80.0)
        self.parser.add_argument('--variance_focus',
                                 type=float,
                                 help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
                                 default=0.85)


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
