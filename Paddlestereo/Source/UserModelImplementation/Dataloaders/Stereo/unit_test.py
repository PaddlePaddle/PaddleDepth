# -*- coding: utf-8 -*-
import argparse
import paddle

try:
    from .stereo_dataset import StereoDataset
except ImportError:
    from stereo_dataset import StereoDataset


class TestFramwork(object):
    def __init__(self, training_list_path: str) -> None:
        super().__init__()
        self._training_list_path = training_list_path

    @staticmethod
    def _init_parser():
        parser = argparse.ArgumentParser(
            description = "The deep learning framework (based on pytorch)")
        parser.add_argument('--imgWidth', type = int, default = 512, help = 'croped width')
        parser.add_argument('--imgHeight', type = int, default = 256, help = 'croped height')
        parser.add_argument('--dataset', type = str, default = 'rob', help = 'dataset')
        parser.add_argument('--dispNum', type = int, default = 256, help = 'dataset')
        parser.add_argument('--startDisp', type = int, default = 0, help = 'dataset')
        args = parser.parse_args()
        return args

    def _init_dataset(self, args: object) -> paddle.io.Dataset:
        dataset = StereoDataset(args, self._training_list_path, True)
        training_dataloader = paddle.io.DataLoader(dataset, batch_size=1,
                                                   shuffle=True, num_workers=2, drop_last=True)
        return training_dataloader

    @staticmethod
    def _check_dataset(training_dataloader: paddle.io.Dataset) -> None:
        for iteration, batch_data in enumerate(training_dataloader):
            print('----------------------------------')
            print('id:', iteration)
            for item in batch_data:
                print(item.shape, type(item))
            print('----------------------------------')

    def start_test(self):
        args = self._init_parser()
        training_dataloader = self._init_dataset(args)
        self._check_dataset(training_dataloader)


def debug_main():
    training_list_path = '/home2/raozhibo/Documents/Programs/PaddleDepth/Datasets/Stereo/middlebury_training_H_list.csv'
    test_framework = TestFramwork(training_list_path)
    test_framework.start_test()


if __name__ == '__main__':
    debug_main()
