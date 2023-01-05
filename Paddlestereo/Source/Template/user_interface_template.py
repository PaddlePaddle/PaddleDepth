# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class NetWorkInferenceTemplate(object):
    __metaclass__ = ABCMeta
    __NETWORK_INFERENCE = None

    def __init__(self):
        pass

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__NETWORK_INFERENCE is None:
            cls.__NETWORK_INFERENCE = object.__new__(cls)
        return cls.__NETWORK_INFERENCE

    @abstractmethod
    def inference(self, args: object) -> object:
        # get model and dataloader
        # return model, dataloader
        pass

    @abstractmethod
    def user_parser(self, parser: object) -> object:
        # parser.add_argument('--phase', default='train', help='train or test')
        # return parser
        pass
