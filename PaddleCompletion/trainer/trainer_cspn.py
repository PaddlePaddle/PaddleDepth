import argparse
import os

import paddle
from paddle import optimizer
from paddle.io import DataLoader
from tqdm import tqdm

from data_loader import NyuDepth
from loss_funcs import WightedL1Loss
from models import UnetCSPN as CSPN
from utils import cspn_utils as utils


def train_epoch(model, data_loader, loss_fn, optim, epoch, logger):
    error_sum_train = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    model.train()
    loss_sum = 0
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, data in tbar:
        step = epoch * len(data_loader) + i

        optim.clear_grad()
        rgb_image = data['rgb']
        sparse_image = data['d']
        targets = data['gt_depth']
        rgbd_image = paddle.concat((rgb_image, sparse_image), 1)
        outputs = model(rgbd_image)
        loss = loss_fn(outputs, targets)
        loss.backward()
        loss_sum += loss.item()
        optim.step()
        # print('Epoch: [{0}][{1}/{2}]\tLoss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))
        error_result = utils.evaluate_error(gt_depth=targets.clone(), pred_depth=outputs.clone())
        for key in error_sum_train.keys():
            error_sum_train[key] += error_result[key]

        logger.write_log(step, error_result, "train")
        logger.add_scalar('train/learning_rate', optim.get_lr(), step)

        if i % 100 == 0:
            pred_img = outputs[0]  # [1,h,w]
            gt_img = targets[0]  # [1,h,w]
            out_img = utils.get_out_img(pred_img[0], gt_img[0])
            logger.write_image("train", out_img, epoch * len(data_loader) + i)
        error_str = f'Train epoch: {epoch}, loss={loss_sum / (i + 1):.4f}'
        tbar.set_description(error_str)

    for key in error_sum_train.keys():
        error_sum_train[key] /= len(data_loader)
    return error_sum_train, loss_sum / len(data_loader)


@paddle.no_grad()
def val_epoch(model, data_loader, loss_fn, epoch, logger):
    error_sum = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    loss_sum = 0
    model.eval()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, data in tbar:
        step = epoch * len(data_loader) + i
        rgb_image = data['rgb']
        sparse_image = data['d']
        targets = data['gt_depth']
        rgbd_image = paddle.concat((rgb_image, sparse_image), 1)
        outputs = model(rgbd_image)
        loss = loss_fn(outputs, targets)
        loss_sum += loss.item()
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)

        pred_img = outputs[0]  # [1,h,w]
        gt_img = targets[0]  # [1,h,w]
        out_img = utils.get_out_img(pred_img[0], gt_img[0])
        logger.write_image("val", out_img, step)
        logger.write_log(step, error_result, "val")

        for key in error_sum.keys():
            error_sum[key] += error_result[key]

        error_str = f'Val epoch: {epoch}, loss={loss_sum / (i + 1):.4f}'
        tbar.set_description(error_str)

    for key in error_sum.keys():
        error_sum[key] /= len(data_loader)
    return error_sum


def train(args):
    # define logger
    logger = utils.Logger(args.log_dir)
    # set device
    paddle.device.set_device(args.device)
    # load data
    train_set = NyuDepth(args.dataset.root, args.dataset.train_split, args.dataset.train_list_file, args.dataset.sample_num)
    val_set = NyuDepth(args.dataset.root, args.dataset.val_split, args.dataset.val_list_file, args.dataset.sample_num)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # define model
    model = CSPN(args.resnet_pretrain)
    model_named_params = [p for _, p in model.named_parameters() if not p.stop_gradient]
    # define loss
    lose_fn = WightedL1Loss()
    # define lr_scheduler and optimizer
    lr_scheduler = optimizer.lr.ReduceOnPlateau(
        learning_rate=args.optimizer.lr,
        mode=args.scheduler.mode,
        factor=args.scheduler.factor,
        patience=args.scheduler.patience,
        min_lr=args.scheduler.min_lr,
        epsilon=args.scheduler.epsilon,
    )
    optim = optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model_named_params,
        weight_decay=args.optimizer.weight_decay,
        momentum=args.optimizer.momentum,
        use_nesterov=args.optimizer.nesterov,
        # dampening=args.dampening ###paddle not support
    )
    # load pretrain model
    start_epoch = 0
    best_error = {
        'MSE': float('inf'), 'RMSE': float('inf'), 'ABS_REL': float('inf'),
        'MAE': float('inf'),
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    if args.pretrain and os.path.exists(args.pretrain):
        try:
            checkpoints = paddle.load(args.pretrain, return_numpy=True)
            model.set_state_dict(checkpoints['model'])
            optim.set_state_dict(checkpoints['optimizer'])
            lr_scheduler.set_state_dict(checkpoints['lr_scheduler'])
            start_epoch = checkpoints['epoch'] + 1
            best_error = checkpoints['val_metrics']
            print(f'load pretrain model from {args.pretrain}')
        except Exception as e:
            print(f"{e} load pretrain model failed")

    # train
    for epoch in range(start_epoch, args.epoch):
        # print(type(args))
        train_metrics, train_loss = train_epoch(model, train_loader, lose_fn, optim, epoch, logger)
        val_metrics = val_epoch(model, val_loader, lose_fn, epoch, logger)

        lr_scheduler.step(train_loss)

        logger.add_scalar('train_epoch/learning_rate', optim.get_lr(), epoch)
        logger.write_log(epoch, train_metrics, "train_epoch")
        logger.write_log(epoch, val_metrics, "val_epoch")

        state = {
            'args': args.get_dict(),
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        if val_metrics['MSE'] < best_error['MSE']:
            best_error = val_metrics
            is_best = True
        else:
            is_best = False

        if epoch % args.interval == 0:
            paddle.save(state, os.path.join(args.save_path, f"checkpoint_{epoch}.pdparams"))
            print(f"save model at epoch {epoch}")
        if is_best:
            paddle.save(state, os.path.join(args.save_path, "model_best.pdparams"))
            print(f"save best model at epoch {epoch} with val_metrics\n{val_metrics}")
