from utils.config import get_config
import argparse


def parse_args():
    parser = argparse.ArgumentParser('Training PaddleCompletion')
    parser.add_argument('--config', '-c', type=str, help="the path of config yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = get_config(args.config)

    if configs.model_name == "fcfrnet":
        from trainer.trainer_fcfrnet import FCFRNet_train as trainer
        #configs.evaluate = None
        assert configs.train_mode in ["dense", "sparse", "photo", "sparse+photo",
                                      "dense+photo"], "train mode only can be \"dense\", \"sparse\", \"photo\", \"sparse+photo\", \"dense+photo\""
        assert configs.val in ["select", "full"], "val mode only can be \"select\", \"full\""
        assert configs.dataset["input_mode"] in ['d', 'rgb', 'rgbd', 'g',
                                                 'gd'], "input mode only can be \'d\', \'rgb\', \'rgbd\', \'g\', \'gd\'"
    elif configs.model_name == "cspn_resnet50_nyu":
        from trainer.trainer_cspn import train as trainer
    elif configs.model_name == "STDNet":
        from trainer.trainer_std import STD_train as trainer
    elif configs.model_name == "guidenet":
        from trainer.trainer_guidenet import GuideNet_train as trainer
    else:
        raise NotImplementedError("model {} is not implemented".format(configs.model_name))
    trainer(configs)


if __name__ == '__main__':
    main()
