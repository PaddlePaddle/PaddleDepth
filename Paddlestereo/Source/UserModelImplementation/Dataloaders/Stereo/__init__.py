# -*- coding: utf-8 -*-
from SysBasic import Switch
from SysBasic import LogHandler as log
from .stereo_dataloader import StereoDataloader


def stereo_dataloaders_zoo(args: object, name: str) -> object:
    for case in Switch(name):
        if case('sceneflow') or case('kitti2012') or case('kitti2015')\
                or case('crestereo') or case('eth3d') or case('rob'):
            log.info("Enter the StereoDataloader")
            dataloader = StereoDataloader(args)
            break
        if case(''):
            dataloader = None
            log.error("The dataloader's name is error!!!")
    return dataloader
