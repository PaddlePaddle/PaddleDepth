# -*- coding: UTF-8 -*-
from SysBasic import Switch
from SysBasic import LogHandler as log

from .build_training_graph import BuildTrainingGraph
from .build_testing_graph import BuildTestingGraph
from .data_handler_manager import DataHandlerManager


# noinspection PyUnresolvedReferences
def graph_selection(args: object, jf_model: object) -> object:
    graph = None
    for case in Switch(args.mode):
        if case('train'):
            log.info("Enter training graph")
            graph = BuildTrainingGraph(args, jf_model)
            break
        if case('test'):
            log.info("Enter testing graph")
            graph = BuildTestingGraph(args, jf_model)
            break
        if case('background'):
            log.info("Enter background graph")
            graph = BuildTestingGraph(args, jf_model)
            break
        if case('online'):
            log.info("Enter online graph")
            break
        if case('reinforcement_learning'):
            log.info("Enter reinforcement learning graph")
            break
        if case(''):
            log.error("The mode's name is error!!!")
    return graph


def dataloader_selection(args: object, jf_dataloader: object) -> object:
    return DataHandlerManager(args, jf_dataloader)
