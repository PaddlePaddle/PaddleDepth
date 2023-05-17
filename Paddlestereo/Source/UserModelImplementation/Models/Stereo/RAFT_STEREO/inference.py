# -*- coding: utf-8 -*-
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim

from Template import ModelHandlerTemplate
from Algorithm import d_1

from .Networks.model import RAFTStereo


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    
class RAFTStereoInterface(ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    MODEL_ID = 0  # only PSMNet
    LEFT_IMG_ID = 0
    RIGHT_IMG_ID = 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        if self.__args.precision == "fp32":
            mixed_precision = False
        else:
            mixed_precision = True
        
        model = RAFTStereo(mixed_precision=mixed_precision)
        params_info = paddle.summary(model, [(1, 3, 256, 512), (1, 3, 256, 512)])
        print(params_info)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        sch = optim.lr.OneCycleLR(max_learning_rate = lr, total_steps = args.maxEpochs*25+100,
            phase_pct=0.01, anneal_strategy='linear', end_learning_rate=1.0e-6) if args.lr_scheduler else None

        new_lr = sch if sch is not None else lr
        clip_grad_norm = nn.ClipGradByNorm(1.0)    
        opt = optim.AdamW(parameters=model[RAFTStereoInterface.MODEL_ID].parameters(), learning_rate=new_lr, weight_decay=1e-5, epsilon=1e-8, grad_clip=clip_grad_norm)

        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int, epoch: str) -> None:
        if self.MODEL_ID == sch_id:
            sch.step()

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        if self.MODEL_ID == model_id:
            image1 = input_data[self.LEFT_IMG_ID]
            image2 = input_data[self.RIGHT_IMG_ID]

            if self.__args.mode == 'train':
                flow_predictions = model(image1, image2, iters=22)
            else:
                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = model(image1, image2, iters=32, test_mode=True)
                flow_up = padder.unpad(flow_up)
                flow_predictions = [-flow_up.squeeze(0)]

        return flow_predictions

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args 
        res = []

        if self.MODEL_ID == model_id:
            for item in output_data:
                gt_flow = label_data[0][..., 0].unsqueeze(1)

                acc, mae = d_1(item, gt_flow)
                # acc, mae = jf.acc.SMAccuracy.d_1(item, label_data[0])
                res.append(acc[1])
                res.append(mae)

        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        flow_loss = 0.0

        flow_preds = output_data
        valid = label_data[0][..., 1]
        flow_gt = label_data[0][..., 0].unsqueeze(1)
        # exlude invalid pixels and extremely large diplacements
        mag = paddle.sum(flow_gt**2, axis=1).sqrt()
        # exclude extremly large displacements
        valid = ((valid >= 0.5) & (mag < 700)).unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not paddle.isinf(flow_gt[valid.astype('bool')]).any()
        loss_gamma = 0.9

        if self.MODEL_ID == model_id:
            n_predictions = len(flow_preds)
            assert n_predictions >= 1
            for i in range(n_predictions):
                assert not paddle.isnan(flow_preds[i]).any() and not paddle.isinf(flow_preds[i]).any()
                # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
                adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
                i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
                i_loss = (flow_preds[i] - flow_gt).abs()
                assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
                flow_loss += i_weight * i_loss[valid.astype('bool')].mean()

        return [flow_loss] 

    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # model.load_state_dict(checkpoint['model_0'], strict=True)
        # jf.log.info("Model loaded successfully")
        return False

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        if self.__args.pretrain == True:
            return True
        else:
            return False

    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        return None
