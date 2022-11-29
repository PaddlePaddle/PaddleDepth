from visualdl import LogWriter

import json
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
from evaluate_depth import evaluate

import logging


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
        if init_parallel: dist.init_parallel_env()
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

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # build model
        self.models, self.parameters_to_train = build_model(self.opt)

        self.model_lr_scheduler = optimizer.lr.StepDecay(self.opt.learning_rate, self.opt.scheduler_step_size, 0.1)
        self.model_optimizer = optimizer.Adam(self.model_lr_scheduler, parameters=self.parameters_to_train)

        if init_parallel:
            for name, model in self.models.items():
                # model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.models[name] = paddle.DataParallel(model, find_unused_parameters=self.opt.find_unused_parameters)

        if self.opt.load_weights_folder is not None and self.opt.load_weights_folder != 'None':
            self.load_model()

        if self.rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.log_path)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        img_ext = '.png' if self.opt.png else '.jpg'
        # ============= 加载数据集 ==============
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))

        # 删除第一帧和最后一帧
        mid_filenames = []
        for name in train_filenames:
            f_str = "{:010d}{}".format(int(name.split(' ')[1]) - 1, img_ext)
            image_path1 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]) + 1, img_ext)
            image_path2 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]), img_ext)
            image_path3 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)
            if os.path.exists(image_path1) and os.path.exists(image_path2) and os.path.exists(image_path3):
                mid_filenames.append(name)
        train_filenames = mid_filenames

        # for self-supervised depth estimation, the val_dataset is not used.
        val_filenames = readlines(fpath.format("val"))
        mid_filenames = []
        for name in val_filenames:
            f_str = "{:010d}{}".format(int(name.split(' ')[1]) - 1, img_ext)
            image_path1 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]) + 1, img_ext)
            image_path2 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]), img_ext)
            image_path3 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)
            if os.path.exists(image_path1) and os.path.exists(image_path2) and os.path.exists(image_path3):
                mid_filenames.append(name)
        val_filenames = mid_filenames

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
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path, is_train=False, img_ext=img_ext)
        val_sampler = paddle.io.DistributedBatchSampler(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            drop_last=True
        )
        self.val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=self.opt.num_workers)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        if self.rank == 0:
            for mode in ["train", "val"]:
                self.writers[mode] = LogWriter(os.path.join(self.log_path, mode))

        if self.rank == 0:
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset),
                                                                                     len(val_dataset)))
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

            # val_loss = self.val()
            abs_rel = evaluate(self.opt, weights_floder)
            self.logger.info(f'In epoch {self.epoch}, the abs_rel is {abs_rel}.')
            if abs_rel < best_val_loss:
                best_val_loss = abs_rel
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
            loss_list.append(losses["loss"].numpy()[0])

            duration = time.time() - last_log_time

            if self.rank == 0 and (batch_idx + 1) % self.opt.log_frequency == 0:
                self.log_time(batch_idx + 1, duration / self.opt.log_frequency, sum(loss_list) / len(loss_list))
                last_log_time = time.time()

            if self.rank == 0 and self.step % self.opt.visualdl_frequency == 0:
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.visualdl_log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        """
        Pass a minibatch through the network and generate images and losses
        """

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = paddle.concat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [paddle.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](paddle.concat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = paddle.concat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = paddle.concat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs


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

                    # 添加评价指标
                    metric_list = []
                    for k in self.depth_metric_names:
                        metric_list.append(losses[k])
                    errors.append(metric_list)
            if "depth_gt" in inputs:
                mean_errors = np.mean(np.array(errors), axis=0)

            if self.rank == 0: self.visualdl_log("val", inputs, outputs, losses)

        val_loss = sum(loss_list) / len(loss_list)

        if "depth_gt" in inputs:
            # 打印信息
            self.logger.info("eval")
            self.logger.info("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            self.logger.info(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            self.logger.info("-> Done!")

        self.logger.info(f'[EVAL:] The validation loss is {val_loss}.')
        return val_loss


    def generate_images_pred(self, inputs, outputs):
        """
        Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # NOTE: Paddle requires img has stop_gradient=False if grid.stop_gradient=False.
                # This should be a bug, already submit an issue to github. 
                # https://github.com/PaddlePaddle/Paddle/issues/38900 
                img = inputs[("color", frame_id, source_scale)].clone()
                img.stop_gradient = False
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    img,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if self.opt.use_depth_hints:
                    if frame_id == 's' and scale == 0:
                        # generate depth hint warped image (only max scale and for stereo image)
                        depth = inputs['depth_hint']
                        cam_points = self.backproject_depth[source_scale](
                            depth, inputs[("inv_K", source_scale)])
                        pix_coords = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T)

                        inputs[("color", frame_id, source_scale)].stop_gradient = False
                        outputs[("color_depth_hint", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords, padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = paddle.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = paddle.concat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = paddle.concat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, paddle.ones(mask.shape))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += paddle.randn(
                    identity_reprojection_loss.shape) * 0.00001

                combined = paddle.concat((identity_reprojection_loss, reprojection_loss), axis=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise = paddle.min(combined, axis=1)
                idxs = paddle.argmin(combined, axis=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).astype(paddle.float32)

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """
        Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = paddle.clip(F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = paddle.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= paddle.median(depth_gt) / paddle.median(depth_pred)

        depth_pred = paddle.clip(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        # mean_errors = np.array(depth_errors)

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

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].numpy().transpose((1, 2, 0)), self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].numpy().transpose((1, 2, 0)), self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    gray2rgb(normalize_image(outputs[("disp", s)][j])).numpy().transpose((1, 2, 0)), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                # elif not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

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
