# -*- coding: utf-8 -*-
import argparse
from Template.user_interface_template import NetWorkInferenceTemplate

import UserModelImplementation.user_define as user_def
from UserModelImplementation.Dataloaders import dataloader_selection
from UserModelImplementation.Models import model_selection


class UserInterface(NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        dataloader = dataloader_selection(args, args.task)
        model = model_selection(args, args.task)
        return model, dataloader

    def user_parser(self, parser: object) -> object:
        parser.add_argument('--startDisp', type=int, default=user_def.START_DISP,
                            help='start disparity')
        parser.add_argument('--dispNum', default=user_def.DISP_NUM,
                            help='disparity number')
        parser.add_argument('--lr_scheduler', type=UserInterface.__str2bool,
                            default=user_def.LR_SCHEDULER, help='use or not use lr scheduler')
        parser.add_argument('--task', type=str, default=user_def.STEREO_MATCHING,
                            help='the name of task')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')
