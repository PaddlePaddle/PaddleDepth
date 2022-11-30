# -*- coding: utf-8 -*-
import paddle
from ._meta_ops import MetaOps


class BuildTestingGraph(MetaOps):
    def __init__(self, args: object, jf_model: object) -> None:
        super().__init__(args, jf_model)

    def exec(self, input_data: list, label_data: list = None, is_training: bool = False) -> list:
        assert label_data is None and not is_training
        outputs_data = []
        with paddle.no_grad():
            for i, model_item in enumerate(self._model):
                output_data = self.user_inference(model_item, input_data, i)
                outputs_data.append(output_data)
        return outputs_data
