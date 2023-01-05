# -*- coding: utf-8 -*
from SysBasic import Switch
from SysBasic import LogHandler as log
from .Stereo import stereo_matching_model_selection


def model_selection(args: object, name: str) -> object:
    for case in Switch(name):
        if case('stereo matching'):
            log.info("Enter the stereo matching model")
            model = stereo_matching_model_selection(args, args.modelName)
            break
        if case(''):
            model = None
            log.error("The model's name is error!!!")
    return model
