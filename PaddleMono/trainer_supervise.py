from visualdl import LogWriter
import json
import logging
from utils import *
import cv2
import numpy as np
import time
import tqdm
import os
import random
import datasets
from model.core import build_model
from model.layers import *
import paddle
import paddle.nn.functional as F
from paddle import optimizer
from paddle.io import DataLoader
import paddle.distributed as dist
from pathlib import Path
from evaluate_depth import evaluate_bts
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class Trainer:
    def __init__(self, options):
        self.opt = options
        setup_seed(self.opt.seed)
        self.rank = paddle.distributed.get_rank()
        init_parallel = self.opt.num_gpus > 1
        if init_parallel: 
            paddle.set_device('gpu')
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # build model
        self.models, self.parameters_to_train = build_model(self.opt)

        if self.opt.load_weights_folder is not None and self.opt.load_weights_folder != 'None':
            self.load_model()

        if self.rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.log_path)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_supervise": datasets.KITTIDepthSuperviseDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        img_ext = '.png' if self.opt.png else '.jpg'
        # load Dataset
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "eigen_{}_files_with_gt.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("test"))
        

        self.batch_size = self.opt.batch_size * max(1, self.opt.num_gpus)

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.batch_size * self.opt.num_epochs


        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path, is_train=True, img_ext=img_ext)
        train_sampler = paddle.io.DistributedBatchSampler(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=self.opt.num_workers,
                                       worker_init_fn=setup_seed)
        
        val_dataset = self.dataset(
            self.opt.data_path.replace("train", "test"), val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path, is_train=False, img_ext=img_ext)
        val_sampler = paddle.io.DistributedBatchSampler(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True
        )
        self.val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=self.opt.num_workers)
        self.val_iter = iter(self.val_loader)


        self.model_lr_scheduler = optimizer.lr.PolynomialDecay(self.opt.learning_rate, decay_steps=self.num_total_steps, end_lr=0.00001, power=0.9, cycle=False, verbose=True)

        self.model_optimizer = optimizer.AdamW(self.model_lr_scheduler, epsilon=self.opt.epsilon, 
                    parameters=[{'params': self.models["encoder"].parameters(), 'weight_decay': self.opt.weight_decay},
                                {'params': self.models["depth"].parameters(), 'weight_decay': 0.}])


        if init_parallel:
            self.model_optimizer = fleet.distributed_optimizer(self.model_optimizer)
            for name, model in self.models.items():
                self.models[name] = fleet.distributed_model(model)


        self.writers = {}
        if self.rank == 0:
            for mode in ["train", "val"]:
                self.writers[mode] = LogWriter(os.path.join(self.log_path, mode))

        if self.rank == 0:
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items\n".format(len(train_dataset)))
            print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))
            self.save_opts()

        if not self.opt.no_ssim:
            self.ssim = SSIM()

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.silog_criterion = silog_loss(variance_focus=self.opt.variance_focus)

    def set_train(self):
        """
        Convert all models to training mode
        """
        if self.opt.freeze_bn: return self.set_eval()
        for m in self.models.values():
            m.train()
            for param in m.parameters():
                param.trainable = True

    def set_eval(self):
        """
        Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
        Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        best_val_loss = float("inf")
        weights_floder = None
        self.start_time = time.time()
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = get_logger(self.log_path + '/train.log')
        self.logger.info('start training!')

        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if self.rank == 0 and (self.epoch + 1) % self.opt.save_frequency == 0:
                weights_floder = self.save_model(self.epoch)

            val_loss = self.val()
            abs_rel = evaluate_bts(self.opt, weights_floder)
            self.logger.info(f'In epoch {self.epoch}, the abs_rel is {abs_rel}.')
            if abs_rel < best_val_loss:
                self.logger.info(f'Save model!!! Best val loss is {abs_rel}.')
                best_val_loss = abs_rel
                self.save_model("best")
                if self.rank == 0: self.save_model("best")

    def run_epoch(self):
        """
        Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        if self.rank == 0: print("Training")
        self.set_train()

        last_log_time = time.time()
        loss_list = []
        for batch_idx, inputs in enumerate(self.train_loader):

            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.clear_grad()
            losses["loss"].backward()

            self.model_optimizer.step()

            if np.isnan(losses["loss"].numpy()[0]):
                print('NaN in loss occurred. Aborting training.')
                return -1

            loss_list.append(losses["loss"].numpy()[0])

            duration = time.time() - last_log_time

            if self.rank == 0 and (batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx + 1, duration / self.opt.log_frequency, sum(loss_list) / len(loss_list))
                last_log_time = time.time()

            if self.rank == 0 and self.step % self.opt.visualdl_frequency == 0:
                self.visualdl_log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        """
        Pass a minibatch through the network and generate images and losses
        """
        # we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features, inputs["focal"])
        mask = inputs["depth_gt"] > 1.0
        mask = paddle.cast(mask, 'bool')
        losses = {}
        loss = self.silog_criterion.forward(outputs["final_depth"], inputs["depth_gt"], mask)
        losses["loss"] = loss

        return outputs, losses


    def val(self, only_val=False):
        """
        Validate the model on the validation set
        """
        self.set_eval()
        if self.rank == 0: print("Validating")
        if only_val:
            os.makedirs(self.log_path, exist_ok=True)
            self.logger = get_logger(self.log_path + '/val.log')
            self.epoch = 0
            self.step = 0

        with paddle.no_grad():
            loss_list = []
            errors = []
            for batch_idx, inputs in enumerate(self.val_loader):
                outputs, losses = self.process_batch(inputs)
                loss_list.append(losses["loss"].numpy()[0])

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                    # add evaluation indicators
                    metric_list = []
                    for k in self.depth_metric_names:
                        metric_list.append(losses[k])
                    errors.append(metric_list)
            if "depth_gt" in inputs:
                mean_errors = np.mean(np.array(errors), axis=0)

            if self.rank == 0: self.visualdl_log("val", inputs, outputs, losses)

        val_loss = sum(loss_list) / len(loss_list)

        if "depth_gt" in inputs:
            self.logger.info("eval")
            self.logger.info("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            self.logger.info(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            self.logger.info("-> Done!")

        self.logger.info(f'[EVAL:] The validation loss is {val_loss}.')
        return val_loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """
        Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth_eval = 1e-3
        max_depth_eval = 80.0
        depth_pred = outputs["final_depth"]

        pred = depth_pred.detach()[0][0]

        depth_gt = inputs["depth_gt"].numpy()[0][0]
        gt_height, gt_width = 352, 704

        top_margin = int(gt_height - 352)
        left_margin = int((gt_width - 704) / 2)
        pred_depth_uncropped = np.zeros((gt_height, gt_width), dtype=np.float32)
        pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 704] = pred
        pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval
        pred_depth[np.isnan(pred_depth)] = min_depth_eval

        eval_mask = np.logical_and(depth_gt > min_depth_eval, depth_gt < max_depth_eval)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(eval_mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(eval_mask, crop_mask)

        depth_gt = paddle.to_tensor(depth_gt[mask])
        depth_pred = paddle.to_tensor(pred_depth[mask])

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i])

    def log_time(self, batch_idx, duration, loss):
        """
        Print a logging statement to the terminal
        """
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        self.logger.info(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                             sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def visualdl_log(self, mode, inputs, outputs, losses):
        """
        Write an event to the VisualDL
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        if mode=='train':
            for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
                for s in self.opt.scales:
                    for frame_id in [0]:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].numpy().transpose((1, 2, 0)), self.step)
                    writer.add_image(
                        "final_depth/{}".format(j),
                        gray2rgb(normalize_image(outputs[("final_depth")][j])).numpy().transpose((1, 2, 0)), self.step)

                    if self.opt.predictive_mask:
                        for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                            writer.add_image(
                                "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                                outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                                self.step)

    def save_opts(self):
        """
        Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, suffix):
        """
        Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(suffix))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pdparams".format(model_name))
            to_save = model.state_dict()
            paddle.save(to_save, save_path)
        return save_folder
        # do not save parameter in the optimizor to save space
        # save_path = os.path.join(save_folder, "{}.pdparams".format("adam"))
        # paddle.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """
        Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, n)
            model_dict = self.models[n].state_dict()
            pretrained_dict = load_weight_file(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = load_weight_file(optimizer_load_path)
            self.model_optimizer.load_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
