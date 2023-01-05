# -*- coding: utf-8 -*-
from SysBasic import Switch
from SysBasic import LogHandler as log
from .Stereo import stereo_dataloaders_zoo


def dataloader_selection(args: object, name: str) -> object:

    for case in Switch(name):
        if case('stereo matching'):
            log.info("Enter the stereo matching dataloader")
            dataloader = stereo_dataloaders_zoo(args, args.dataset)
            break
        if case(''):
            dataloader = None
            log.error("The model's name is error!!!")
    return dataloader
