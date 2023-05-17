# -*- coding: utf-8 -*
from SysBasic import Switch
from SysBasic import LogHandler as log
from .PSMNet import PSMNetInterface
from .PWCNet import PWCMNetInterface
from .RAFT_STEREO import RAFTStereoInterface


def stereo_matching_model_selection(args: object, name: str) -> object:
    for case in Switch(name):
        if case('PSMNet'):
            log.info("Enter the PSMNet model")
            model = PSMNetInterface(args)
            break
        if case('PWCNet'):
            log.info("Enter the PWCNet model")
            model = PWCMNetInterface(args)
            break
        if case('RAFT_STEREO'):
            log.info("Enter the RAFT_STEREO model")
            model = RAFTStereoInterface(args)
            break
        if case(''):
            model = None
            log.error("The model's name is error!!!")
    return model
