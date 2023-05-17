from utils.config import get_config
import argparse


def parse_args():
    parser = argparse.ArgumentParser('PaddleCompletion Evaluation')
    parser.add_argument('--config', '-c', type=str, help="the path of config yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = get_config(args.config)

    if configs.model_name == "fcfrnet":
        assert configs.evaluate is not None, "please set evaluate in config file"
        from trainer.trainer_fcfrnet import FCFRNet_train as evaluator
        assert configs.train_mode in ["dense", "sparse", "photo", "sparse+photo",
                                      "dense+photo"], "train mode only can be \"dense\", \"sparse\", \"photo\", \"sparse+photo\", \"dense+photo\""
        assert configs.val in ["select", "full"], "val mode only can be \"select\", \"full\""
        assert configs.dataset["input_mode"] in ['d', 'rgb', 'rgbd', 'g',
                                                 'gd'], "input mode only can be \'d\', \'rgb\', \'rgbd\', \'g\', \'gd\'"
    elif configs.model_name == "cspn_resnet50_nyu":
        from evulator.evulator_cspn import main as evaluator
    elif configs.model_name == "guidenet":
        from trainer.trainer_guidenet import GuideNet_train as evaluator
    elif configs.model_name == "STDNet":
        from trainer.trainer_std import STD_train as evaluator
    else:
        raise NotImplementedError("model {} is not implemented".format(configs.model_name))
    evaluator(configs)


if __name__ == '__main__':
    main()
