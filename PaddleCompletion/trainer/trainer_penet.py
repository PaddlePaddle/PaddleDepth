import os
import time

from paddle import io
from paddle import optimizer as optim
import paddle
from utils import helper
from utils.metric import AverageMeter, Result
from loss_funcs.MaskedMSELoss import MaskedMSELoss
from data_loader.kitti_loader import KittiDepth
from models import PENet_C1, PENet_C2, PENet_C4, PENet_C1_train, PENet_C2_train, ENet

multi_batch_size = 1


def adjust_learning_rate(lr_init, optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init
    if args.model["network_model"] == "pe" and args.model["freeze_backbone"] is False:
        if epoch >= 10:
            lr = lr_init * 0.5
        if epoch >= 20:
            lr = lr_init * 0.1
        if epoch >= 30:
            lr = lr_init * 0.01
        if epoch >= 40:
            lr = lr_init * 0.0005
        if epoch >= 50:
            lr = lr_init * 1e-05
    else:
        if epoch >= 10:
            lr = lr_init * 0.5
        if epoch >= 15:
            lr = lr_init * 0.1
        if epoch >= 25:
            lr = lr_init * 0.01
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
    optimizer.set_lr(lr)
    return lr


def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias
    block_average_meter = AverageMeter()
    block_average_meter.reset()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    depth_criterion = MaskedMSELoss()
    assert mode in [
        "train",
        "val",
        "eval",
        "test_prediction",
        "test_completion",
    ], "unsupported mode: {}".format(mode)
    if mode == "train":
        model.train()
        lr = adjust_learning_rate(args.optimizer["lr"], optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0
    paddle.device.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        dstart = time.time()
        gt = (
            batch_data["gt_depth"]
            if mode != "test_prediction" and mode != "test_completion"
            else None
        )
        data_time = time.time() - dstart
        pred = None
        start = None
        gpu_time = 0
        if args.model["network_model"] == "e":
            start = time.time()
            st1_pred, st2_pred, pred = model(batch_data)
        else:
            start = time.time()
            pred = model(batch_data)
        if args.evaluate:
            gpu_time = time.time() - start
        depth_loss, photometric_loss = 0, 0
        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2 = 1, 3
        if actual_epoch <= round1:
            w_st1, w_st2 = 0.2, 0.2
        elif actual_epoch <= round2:
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0
        if mode == "train":
            depth_loss = depth_criterion(pred, gt)
            if args.model["network_model"] == "e":
                st1_loss = depth_criterion(st1_pred, gt)
                st2_loss = depth_criterion(st2_pred, gt)
                loss = (
                    (1 - w_st1 - w_st2) * depth_loss
                    + w_st1 * st1_loss
                    + w_st2 * st2_loss
                )
            else:
                loss = depth_loss
            if i % multi_batch_size == 0:
                optimizer.clear_grad()
            loss.backward()
            if i % multi_batch_size == multi_batch_size - 1 or i == len(loader) - 1:
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))
        if not args.evaluate:
            gpu_time = time.time() - start
        with paddle.no_grad():
            mini_batch_size = next(iter(batch_data.values())).shape[0]
            result = Result()
            if mode != "test_prediction" and mode != "test_completion":
                result.evaluate(pred, gt, pred, pred, photometric_loss)
                [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
                if mode != "train":
                    logger.conditional_print(
                        mode,
                        i,
                        epoch,
                        lr,
                        len(loader),
                        block_average_meter,
                        average_meter,
                    )
                logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
                logger.conditional_save_pred(mode, i, pred, epoch)
    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not mode == "train":
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)
    return avg, is_best


def PENet_train(args):
    args.use_pose = "photo" in args.train_mode
    # args.pretrained = not args.no_pretrained
    args.result = os.path.join("", "results")
    args.use_rgb = ("rgb" in args.dataset["input_mode"]) or args.use_pose
    args.use_d = "d" in args.dataset["input_mode"]
    args.use_g = "g" in args.dataset["input_mode"]

    args.val_h = 352
    args.val_w = 1216

    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate), end="")
            checkpoint = paddle.load(path=args.evaluate)
            args.start_epoch = checkpoint["epoch"] + 1
            args.data_folder = args_new.dataset["data_folder"]
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
    elif args.resume:
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume), end="")
            checkpoint = paddle.load(path=args.resume)
            args.start_epoch = checkpoint["epoch"] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return
    print("=> creating model and optimizer ... ", end="")
    model = None
    penet_accelerated = False
    if args.model["network_model"] == "e":
        model = ENet(args)
    elif is_eval is False:
        if args.model["dilation_rate"] == 1:
            model = PENet_C1_train(args)
        elif args.model["dilation_rate"] == 2:
            model = PENet_C2_train(args)
        elif args.model["dilation_rate"] == 4:
            model = PENet_C4(args)
            penet_accelerated = True
    elif args.model["dilation_rate"] == 1:
        model = PENet_C1(args)
        penet_accelerated = True
    elif args.model["dilation_rate"] == 2:
        model = PENet_C2(args)
        penet_accelerated = True
    elif args.model["dilation_rate"] == 4:
        model = PENet_C4(args)
        penet_accelerated = True
    if penet_accelerated is True:
        model.encoder3.stop_gradient = True
        model.encoder5.stop_gradient = True
        model.encoder7.stop_gradient = True
    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None

    if checkpoint is not None:
        if args.model["freeze_backbone"] is True:
            model.backbone.set_state_dict(checkpoint["model"])
        else:
            model.set_state_dict(checkpoint["model"])

        print("=> checkpoint state loaded.")

    logger = helper.logger(args)
    print("=> logger created.")
    val_dataset = KittiDepth("val", args)
    val_loader = io.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    print("\t==> val_loader size:{}".format(len(val_loader)))
    if is_eval is True:
        for p in model.parameters():
            p.stop_gradient = True
        _, is_best = iterate(
            "val", args, val_loader, model, None, logger, args.start_epoch - 1
        )
        return
    if args.model["freeze_backbone"] is True:
        for p in model.backbone.parameters():
            p.stop_gradient = True
        model_named_params = [
            p for _, p in model.named_parameters() if not p.stop_gradient
        ]
        optimizer = optim.Adam(
            parameters=model_named_params,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            beta1=0.9,
            beta2=0.99,
        )
    elif args.model["network_model"] == "pe":
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if not p.stop_gradient
        ]
        model_new_params = [
            p for _, p in model.named_parameters() if not p.stop_gradient
        ]
        model_new_params = list(set(model_new_params) - set(model_bone_params))
        optimizer = optim.Adam(
            parameters=[
                {
                    "params": model_bone_params,
                    "learning_rate": args.optimizer["lr"] / 10,
                },
                {"params": model_new_params},
            ],
            learning_rate=args.optimizer["lr"],
            weight_decay=args.optimizer["weight_decay"],
            beta1=0.9,
            beta2=0.99,
        )
    else:
        model_named_params = [
            p for _, p in model.named_parameters() if not p.stop_gradient
        ]
        optimizer = optim.Adam(
            parameters=model_named_params,
            learning_rate=args.optimizer["lr"],
            weight_decay=args.optimizer["weight_decay"],
            beta1=0.9,
            beta2=0.99,
        )
    print("completed.")

    model = paddle.DataParallel(model)
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth("train", args)
        train_loader = io.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        )
        print("\t==> train_loader size:{}".format(len(train_loader)))
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)
        for p in model.parameters():
            p.stop_gradient = True
        _, is_best = iterate("val", args, val_loader, model, None, logger, epoch)
        for p in model.parameters():
            p.stop_gradient = False
        if args.model["freeze_backbone"] is True:
            for p in model._layers.backbone.parameters():
                p.stop_gradient = True
        if penet_accelerated is True:
            model.encoder3.stop_gradient = True
            model.encoder5.stop_gradient = True
            model.encoder7.stop_gradient = True
        helper.save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args.get_dict(),
            },
            is_best,
            epoch,
            logger.output_directory,
        )
