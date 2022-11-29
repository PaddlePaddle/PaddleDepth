import argparse
from trainer import Trainer
from utils.workspace import load_config, merge_config
import paddle

class ValOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="options")
        # CFG
        self.parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
        self.parser.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")
        
        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory")

        # TRAINING options
        self.parser.add_argument('--num_gpus',
                                 type=int,
                                 help='number of gpus used in training')
        self.parser.add_argument("--seed",
                                 type=int,
                                 help='seed used in training.')
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_full", "eigen_lite"],
                                 default="eigen_full")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.001)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the losses",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 type=bool,
                                 default=True)
                                 #action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load")
        # DEPTH HINT options
        self.parser.add_argument("--use_depth_hints",
                                 help="if set, apply depth hints during training",
                                 type=bool)

        self.parser.add_argument("--depth_hint_path",
                                 help="path to load precomputed depth hints from. If not set will"
                                      "be assumed to be data_path/depth_hints",
                                 type=str)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate")
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="number of epochs")
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs")
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler")

        # ABLATION options
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch")

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers")

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


def main(opts):
    # 读取配置文件
    if not opts.cfg:
        raise RuntimeError('No configuration file specified.')
    cfg = load_config(opts.cfg)
    # 用命令行更新配置文件
    if opts.data_path is not None:
        cfg['data_path'] = opts.data_path
    if opts.split is not None:
        cfg['split'] = opts.split
    if opts.num_gpus is not None:
        cfg['num_gpus'] = opts.num_gpus
    if opts.seed is not None:
        cfg['seed'] = opts.seed
    if opts.height is not None:
        cfg['height'] = opts.height
    if opts.width is not None:
        cfg['width'] = opts.width
    if opts.num_layers is not None:
        cfg['num_layers'] = opts.num_layers
    if opts.disparity_smoothness is not None:
        cfg['disparity_smoothness'] = opts.disparity_smoothness
    if opts.model_name is not None:
        cfg['model_name'] = opts.model_name
    if opts.num_workers is not None:
        cfg['num_workers'] = opts.num_workers
    if opts.batch_size is not None:
        cfg['batch_size'] = opts.batch_size
    if opts.load_weights_folder is not None:
        cfg['load_weights_folder'] = opts.load_weights_folder
    if opts.models_to_load is not None:
        cfg['models_to_load'] = opts.models_to_load
    if opts.frame_ids is not None:
        cfg['frame_ids'] = opts.frame_ids
    
    merge_config(opts.opt)
    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'
    print(cfg)
    # 执行
    trainer = Trainer(cfg)
    trainer.val(only_val=True)

if __name__ == "__main__":
    options = ValOptions()
    opts = options.parse()
    main(opts)
    