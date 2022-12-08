# -*- coding: utf-8 -*-
import paddle
import paddle.fluid as fluid

from Template import ModelHandlerTemplate
from Algorithm import d_1

from .lr_scheduler import StereoLRScheduler
from .Networks.model import PSMNet


class PSMNetInterface(ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    MODEL_ID = 0  # only PSMNet
    LEFT_IMG_ID = 0
    RIGHT_IMG_ID = 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        args = self.__args
        model = PSMNet(args.dispNum)
        # params_info = paddle.summary(model, [(1, 3, 256, 512), (1, 3, 256, 512)])
        # print(params_info)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = paddle.optimizer.Adam(learning_rate=lr,
                                    parameters=model[PSMNetInterface.MODEL_ID].parameters())

        if args.lr_scheduler:
            sch = StereoLRScheduler(lr, [200, 300])
        else:
            sch = None

        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int, epoch: str) -> None:
        if self.MODEL_ID == sch_id:
            sch.step(epoch)

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        pred1, pred2, pred3 = None, None, None
        res = []
        if self.MODEL_ID == model_id:
            if self.__args.mode == 'train':
                pred1, pred2, pred3 = model(input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID])
                res = [pred1, pred2, pred3]
            else:
                pred3 = model(input_data[self.LEFT_IMG_ID], input_data[self.RIGHT_IMG_ID])
                res = [pred3]

        return res

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        res = []

        if self.MODEL_ID == model_id:
            for item in output_data:
                acc, mae = d_1(item, label_data[0])
                # acc, mae = jf.acc.SMAccuracy.d_1(item, label_data[0])
                res.append(acc[1])
                res.append(mae)

        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args
        total_loss = None
        res = []

        if self.MODEL_ID == model_id:
            for item in output_data:
                mask = (label_data[0] > args.startDisp
                        ) & (label_data[0] < args.startDisp + args.dispNum)
                total_num = mask.sum() + 1e-10
                loss = fluid.layers.smooth_l1(item * mask, label_data[0] * mask)
                loss = loss / total_num
                # loss = jf.loss.SMLoss.smooth_l1(item, label_data[0],
                #                                args.startDisp,
                #                                args.startDisp + args.dispNum)
                loss = paddle.fluid.layers.reduce_mean(loss, dim=0)
                res.append(loss)
            total_loss = sum(res)

        return [total_loss] + res

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
        # opt.load_state_dict(checkpoint['optimizer'])
        # jf.log.info("Model loaded successfully")
        return False

    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        return None
