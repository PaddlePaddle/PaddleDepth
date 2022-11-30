# -*- coding: utf-8 -*-
import argparse
import SysBasic.define as sys_define


# Parse the train model's para
class ArgsParser(object):
    """docstring for ArgsParser"""

    def __init__(self):
        super().__init__()

    def parse_args(self, info: str, user_define_func: object = None) -> object:
        parser = argparse.ArgumentParser(
            description="The deep learning framework (based on pytorch) - " + info)
        parser = self.__program_setting(parser)
        parser = self.__path_setting(parser)
        parser = self.__training_setting(parser)
        parser = self.__img_setting(parser)
        parser = self.__user_setting(parser)
        parser = self.__load_user_define(parser, user_define_func)
        return parser.parse_args()

    # noinspection PyCallingNonCallable
    @staticmethod
    def __load_user_define(parser: object, user_define_func: object) -> object:
        if user_define_func is not None:
            user_parser = user_define_func(parser)
            if isinstance(user_parser, type(parser)):
                parser = user_parser
        return parser

    @staticmethod
    def __program_setting(parser: object) -> object:
        parser.add_argument('--mode', default='train',
                            help='train or test')
        parser.add_argument('--gpu', type=int, default=sys_define.GPU_NUM,
                            help='state the num of gpu: 0, 1, 2 or 3 ...')
        parser.add_argument('--auto_save_num', type=int, default=sys_define.AUTO_SAVE_NUM,
                            help='AUTO_SAVE_NUM')
        parser.add_argument('--dataloaderNum', type=int, default=sys_define.DATA_LOADER_NUM,
                            help='the number of dataloader')
        parser.add_argument('--pretrain', default=False, type=ArgsParser.__str2bool,
                            help='true or false')
        parser.add_argument('--ip', default=sys_define.IP,
                            help='ip')
        parser.add_argument('--port', default=sys_define.PORT,
                            help='port')
        parser.add_argument('--dist', default=sys_define.DIST, type=ArgsParser.__str2bool,
                            help='use DDP or DP')
        return parser

    @staticmethod
    def __path_setting(parser: object) -> object:
        parser.add_argument('--trainListPath', default=sys_define.TRAIN_LIST_PATH,
                            help='training list path or testing list path')
        parser.add_argument('--valListPath', default=sys_define.VAL_LIST_PATH,
                            help='val list path')
        parser.add_argument('--outputDir', default=sys_define.DATA_OUTPUT_PATH,
                            help="The output's path. e.g. './result/'")
        parser.add_argument('--modelDir', default=sys_define.MODEL_PATH,
                            help="The model's path. e.g. ./model/")
        parser.add_argument('--resultImgDir', default=sys_define.RESULT_OUTPUT_PATH,
                            help="The save path. e.g. ./ResultImg/")
        parser.add_argument('--log', default=sys_define.LOG_OUTPUT_PATH,
                            help="the log file")
        return parser

    @staticmethod
    def __training_setting(parser: object) -> object:
        parser.add_argument('--sampleNum', type=int, default=sys_define.SAMPLE_NUM,
                            help='the number of sample')
        parser.add_argument('--batchSize', type=int, default=sys_define.BATCH_SIZE,
                            help='Batch Size')
        parser.add_argument('--lr', type=float, default=sys_define.LEARNING_RATE,
                            help="Learning rate. e.g. 0.01, 0.001, 0.0001")
        parser.add_argument('--maxEpochs', type=int, default=sys_define.MAX_EPOCHS,
                            help="Max step. e.g. 500")

        return parser

    @staticmethod
    def __img_setting(parser: object) -> object:
        parser.add_argument('--imgWidth', default=sys_define.IMAGE_WIDTH, type=int,
                            help="Image's width. e.g. 512, In the training process is Clipped size")
        parser.add_argument('--imgHeight', default=sys_define.IMAGE_HEIGHT, type=int,
                            help="Image's width. e.g. 256, In the training process is Clipped size")
        parser.add_argument('--imgNum', default=sys_define.IMG_NUM, type=int,
                            help="The number of training images")
        parser.add_argument('--valImgNum', default=sys_define.VAL_IMG_NUM, type=int,
                            help="The number of val images")
        return parser

    @staticmethod
    def __user_setting(parser: object) -> object:
        parser.add_argument('--modelName', default=sys_define.MODEL_NAME,
                            help='model name')
        parser.add_argument('--dataset', default=sys_define.DATASET_NAME,
                            help="the dataset's name")
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            res = True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            res = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return res
