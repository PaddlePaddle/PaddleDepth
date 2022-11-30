import paddle
import time
import os

from utils.metric import AverageMeter, Result
from loss_funcs.MaskedMSELoss import MaskedMSELoss
from loss_funcs.PhotometricLoss import PhotometricLoss
from loss_funcs.SmoothnessLoss import SmoothnessLoss
from utils.helper import adjust_learning_rate, multiscale, logger, save_checkpoint
from utils.inverse_warp import homography_from, Intrinsics
from data_loader.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from models import DepthCompletionNet


def iterate(mode, args, loader, model, optimize, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = adjust_learning_rate(args.lr, optimize, epoch)
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {
            key: val
            for key, val in batch_data.items() if val is not None
        }
        # print(batch_data.keys())
        batch_data['d'][batch_data['d'] > 85] = 85
        batch_data['d'] /= 85.0
        batch_data["rgb"] /= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max())
        gt = batch_data['gt_depth'] if mode != 'test_prediction' and mode != 'test_completion' else None

        data_time = time.time() - start

        start = time.time()
        pred = model(batch_data)  #################
        # pred=batch_data['d']
        # pred*=85
        #     #print(pred.max(),pred.min())
        #     #print(pred.max(), depth_pred.max(), lidar_pred.max(), global_features.max())
        batch_data['d'] *= 85.0
        batch_data['rgb'] *= 255.0
        #     #print(batch_data['d'].max(), batch_data['rgb'].max(), gt.max())
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = MaskedMSELoss(pred, batch_data['d'])
                # mask = (batch_data['d'] < 1e-3).float()
                mask = paddle.to_tensor(batch_data['d'] < 1e-3, dtype="float32")
            elif 'dense' in args.train_mode:
                # res_gt = gt-batch_data['d']
                # res_gt[gt==0]=0
                # if i==0:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+10*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # elif i==1:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+2*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # else:
                depth_loss = MaskedMSELoss(pred, gt)  # +depth_criterion(global_features, res_gt)
                # depth_loss = depth_criterion(pred, gt)+0.1*depth_criterion(depth_pred, gt)+0.1*depth_criterion(lidar_pred, gt)+0.1*depth_criterion(global_features, res_gt)
                # mask = (gt < 1e-3).float()
                mask = paddle.to_tensor((gt < 1e-3), dtype="float32")
            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = multiscale(pred)
                rgb_curr_array = multiscale(batch_data['rgb'])
                rgb_near_array = multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    photometric_loss += PhotometricLoss(
                        rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = SmoothnessLoss(pred) if args.w2 > 0 else 0

            # backprop
            loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss

            optimize.clear_grad()
            loss.backward()
            optimize.step()

        gpu_time = time.time() - start
        if mode != 'train':
            pred[pred < 0.9] = 0.9
            pred[pred > 85] = 85
            # pred[depth_pred<0.9] = 0.9
            # pred[depth_pred>85] = 85
            # pred[lidar_pred<0.9] = 0.9
            # pred[lidar_pred>85] = 85
        # measure accuracy and record loss
        with paddle.no_grad():

            mini_batch_size = next(iter(batch_data.values())).shape[0]
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.detach(), gt.detach(), pred.detach(), pred.detach(), photometric_loss)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]
            logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)

            logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)
        # if mode=='train':
        #    break

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)
    return avg, is_best


@paddle.no_grad()
def iterate_val(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {
            key: val
            for key, val in batch_data.items() if val is not None
        }

        batch_data['d'][batch_data['d'] > 85] = 85
        batch_data['d'] /= 85.0
        batch_data['rgb'] /= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max())
        gt = batch_data[
            'gt_depth'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start

        start = time.time()
        pred = model(batch_data)
        # pred=batch_data
        # print(pred.max(),pred.min())
        # print(pred.max(), depth_pred.max(), lidar_pred.max(), global_features.max())
        batch_data['d'] *= 85.0
        batch_data['rgb'] *= 255.0
        # print(batch_data['d'].max(), batch_data['rgb'].max(), gt.max())
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = MaskedMSELoss(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                # res_gt = gt-batch_data['d']
                # res_gt[gt==0]=0
                # if i==0:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+10*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # elif i==1:
                #    depth_loss = depth_criterion(pred, gt)+2*depth_criterion(depth_pred, gt)+2*depth_criterion(lidar_pred, gt)#+depth_criterion(global_features, res_gt)
                # else:
                depth_loss = MaskedMSELoss(pred, gt)  # +depth_criterion(global_features, res_gt)
                # depth_loss = depth_criterion(pred, gt)+0.1*depth_criterion(depth_pred, gt)+0.1*depth_criterion(lidar_pred, gt)+0.1*depth_criterion(global_features, res_gt)
                mask = (gt < 1e-3).float()

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = multiscale(pred)
                rgb_curr_array = multiscale(batch_data['rgb'])
                rgb_near_array = multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    if args.use_pose:
                        # hard-coded KITTI camera intrinsics
                        K = load_calib(args.dataset['calib_path'])
                        fu, fv = float(K[0, 0]), float(K[1, 1])
                        cu, cv = float(K[0, 2]), float(K[1, 2])
                        kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    photometric_loss += PhotometricLoss(
                        rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = SmoothnessLoss(pred) if args.w2 > 0 else 0

            # backprop
            loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start
        if mode != 'train':
            pred[pred < 0.9] = 0.9
            pred[pred > 85] = 85
            # depth_pred[depth_pred<0.9] = 0.9
            # depth_pred[depth_pred>85] = 85
            # lidar_pred[lidar_pred<0.9] = 0.9
            # lidar_pred[lidar_pred>85] = 85
        # measure accuracy and record loss
        with paddle.no_grad():
            mini_batch_size = next(iter(batch_data.values())).shape[0]
            # print(mini_batch_size)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.detach(), gt.detach(), pred.detach(), pred.detach(), photometric_loss)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]

            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)
        # if mode=='train' and i==100:
        #    break

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


def FCFRNet_train(args):
    args.use_pose = ("photo" in args.train_mode)
    # args.pretrained = not args.no_pretrained
    args.result = os.path.join('', 'results')
    args.use_rgb = ('rgb' in args.dataset["input_mode"]) or args.use_pose
    args.use_d = 'd' in args.dataset["input_mode"]
    args.use_g = 'g' in args.dataset["input_mode"]
    if args.use_pose:
        args.w1, args.w2 = 0.1, 0.1
    else:
        args.w1, args.w2 = 0, 0
    print(args)

    if args.use_pose:
        # hard-coded KITTI camera intrinsics
        K = load_calib(args.dataset['calib_path'])
        fu, fv = float(K[0, 0]), float(K[1, 1])
        cu, cv = float(K[0, 2]), float(K[1, 2])
        kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)

    checkpoint = None
    is_eval = False

    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')

            checkpoint = paddle.load(args.evaluate)
            args = args
            args.data_folder = args_new.dataset['data_folder']
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    if args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return
    model = DepthCompletionNet(args)
    model_named_params = [
        p for _, p in model.named_parameters() if not p.stop_gradient
    ]
    optim = paddle.optimizer.Adam(learning_rate=args.optimizer['lr'], parameters=model_named_params,
                                  weight_decay=args.optimizer['weight_decay'], )
    print("completed.")
    if checkpoint is not None:
        # params_state = paddle.load(path=checkpoint)
        model.set_dict(checkpoint)
        # optim.load(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = paddle.DataParallel(model)
    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers)
        print("\t==> train_loader size:{}".format(len(train_loader)))

    val_dataset = KittiDepth('val', args)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    _logger = logger(args)
    # if checkpoint is not None:
    #     logger.best_result = checkpoint['best_result']
    # print("=> logger created.")

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, _logger, args.epochs)
        return

    # main loop
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optim, _logger, epoch)  # train for one epoch
        result, is_best = iterate_val("val", args, val_loader, model, None, _logger,
                                      epoch)  # evaluate on validation set
        save_checkpoint({  # save checkpoint
            'epoch': epoch,
            'model': model.state_dict(),
            'best_result': _logger.best_result,
            'optimizer': optim.state_dict(),
            'args': args.get_dict(),
        }, is_best, epoch, _logger.output_directory)
