import os

import cv2
import paddle
from paddle.io import DataLoader
from tqdm import tqdm

from data_loader import NyuDepth
from loss_funcs import WightedL1Loss
from models import UnetCSPN as CSPN
from utils import cspn_utils as utils


@paddle.no_grad()
def test_vis_epoch(model, data_loader, loss_fn, epoch, logger,args):
    error_sum = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    model.eval()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, data in tbar:
        rgb_image = data['rgb']
        sparse_image = data['d']
        targets = data['gt_depth']
        rgbd_image = paddle.concat((rgb_image, sparse_image), 1)
        outputs = model(rgbd_image)
        loss = loss_fn(outputs, targets).item()
        tbar.set_description(f'Item {i} | Loss {loss:.4f}')
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        outputs = outputs.numpy()
        targets = targets.numpy()

        pred_img = outputs[0]  # [1,h,w]
        gt_img = targets[0]  # [1,h,w]

        out_img = utils.get_out_img(pred_img[0], gt_img[0])
        cv2.imwrite(f'{args.out_path}/result_{i}.png', out_img)
        logger.write_image("val", out_img, epoch * len(data_loader) + i)

        for key in error_sum.keys():
            error_sum[key] += error_result[key][0]

        logger.write_log(epoch * len(data_loader) + i, error_result, "test")

    for key in error_sum.keys():
        error_sum[key] /= len(data_loader)
    return error_sum


def main(args):
    logger = utils.Logger(args.log_dir)
    paddle.device.set_device(args.device)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    val_set = NyuDepth(args.dataset.root, args.dataset.val_split, args.dataset.val_list_file, args.dataset.sample_num)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    model = CSPN(pretrained=False)
    if os.path.exists(args.model_path):
        params = paddle.load(args.model_path, return_numpy=True)
        model.set_state_dict(params['model'])
        print(f'load model from {args.model_path}')
    else:
        raise RuntimeError(f'no model found at {args.model_path}')

    lose_fn = WightedL1Loss()
    val_metrics = test_vis_epoch(model, val_loader, lose_fn, 0, logger, args)
    for key in val_metrics.keys():
        print(f'{key}: {val_metrics[key]:.4f}')
    print(val_metrics)
