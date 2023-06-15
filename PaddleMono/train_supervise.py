from trainer_supervise import Trainer
from options import TrainOptions
from utils.workspace import load_config, merge_config
import paddle

if __name__ == "__main__":
    options = TrainOptions()
    opts = options.parse()
    if not opts.cfg:
        raise RuntimeError('No configuration file specified.')
    cfg = load_config(opts.cfg)

    if opts.data_path is not None:
        cfg['data_path'] = opts.data_path
    if opts.log_dir is not None:
        cfg['log_dir'] = opts.log_dir
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
    if opts.use_stereo:
        cfg['use_stereo'] = opts.use_stereo
    if opts.use_depth_hints:
        cfg['use_depth_hints'] = opts.use_depth_hints
    if opts.png:
        cfg['png'] = opts.png
    if opts.freeze_bn:
        cfg['freeze_bn'] = opts.freeze_bn
    if opts.depth_hint_path is not None:
        cfg['depth_hint_path'] = opts.depth_hint_path
    if opts.weights_init is not None:
        cfg['weights_init'] = opts.weights_init
    if opts.scheduler_step_size is not None:
        cfg['scheduler_step_size'] = opts.scheduler_step_size
    if opts.model_name is not None:
        cfg['model_name'] = opts.model_name
    if opts.num_workers is not None:
        cfg['num_workers'] = opts.num_workers
    if opts.batch_size is not None:
        cfg['batch_size'] = opts.batch_size
    if opts.learning_rate is not None:
        cfg['learning_rate'] = opts.learning_rate
    if opts.start_epoch is not None:
        cfg['start_epoch'] = opts.start_epoch
    if opts.num_epochs is not None:
        cfg['num_epochs'] = opts.num_epochs
    if opts.scheduler_step_size is not None:
        cfg['scheduler_step_size'] = opts.scheduler_step_size
    if opts.load_weights_folder is not None:
        cfg['load_weights_folder'] = opts.load_weights_folder
    if opts.models_to_load is not None:
        cfg['models_to_load'] = opts.models_to_load
    if opts.log_frequency is not None:
        cfg['log_frequency'] = opts.log_frequency
    if opts.save_frequency is not None:
        cfg['save_frequency'] = opts.save_frequency
    if opts.visualdl_frequency is not None:
        cfg['visualdl_frequency'] = opts.visualdl_frequency
    if opts.eval_mono:
        cfg['eval_mono'] = opts.eval_mono
    if opts.eval_stereo:
        cfg['eval_stereo'] = opts.eval_stereo
    cfg['ext_disp_to_eval'] = opts.ext_disp_to_eval
    cfg['eval_out_dir'] = opts.eval_out_dir
    if opts.eval_split is not None:
        cfg['eval_split'] = opts.eval_split
    if opts.post_process:
        cfg['post_process'] = opts.post_process
    if opts.save_pred_disps:
        cfg['save_pred_disps'] = opts.save_pred_disps
    if opts.no_eval:
        cfg['no_eval'] = opts.no_eval
    if opts.eval_eigen_to_benchmark:
        cfg['eval_eigen_to_benchmark'] = opts.eval_eigen_to_benchmark
    if opts.disable_median_scaling:
        cfg['disable_median_scaling'] = opts.disable_median_scaling
    # for BTS
    if opts.encoder is not None:
        cfg['encoder'] = opts.encoder
    if opts.max_depth is not None:
        cfg['max_depth'] = opts.max_depth
    if opts.variance_focus is not None:
        cfg['variance_focus'] = opts.variance_focus
    if opts.epsilon is not None:
        cfg['epsilon'] = opts.epsilon
    if opts.weight_decay is not None:
        cfg['weight_decay'] = opts.weight_decay
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

    # # FIXME: Temporarily solve the priority problem of FLAGS.opt
    # merge_config(opts.opt)
    # check.check_config(cfg)
    # check.check_gpu(cfg.use_gpu)
    # check.check_npu(cfg.use_npu)
    # check.check_version()

    print(cfg)
    trainer = Trainer(cfg)
    trainer.train()