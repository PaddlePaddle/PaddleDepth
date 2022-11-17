#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.regularizer import L1Decay, L2Decay

from ..utils.registry import Registry

LRSCHEDULERS = Registry("LRSCHEDULER")
OPTIMIZERS = Registry("OPTIMIZER")


def build_lr_scheduler(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    return LRSCHEDULERS.get(name)(**cfg_)


def build_optimizer(cfg, lr_scheduler, parameters=None):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    if cfg_.get('weight_decay'):
        if isinstance(cfg_.get('weight_decay'),
                      float):  # just an float factor
            cfg_['weight_decay'] = cfg_.get('weight_decay')
        elif 'L1' in cfg_.get('weight_decay').get(
                'name').upper():  # specify L2 wd and it's float factor
            cfg_['weight_decay'] = L1Decay(
                cfg_.get('weight_decay').get('value'))
        elif 'L2' in cfg_.get('weight_decay').get(
                'name').upper():  # specify L1 wd and it's float factor
            cfg_['weight_decay'] = L2Decay(
                cfg_.get('weight_decay').get('value'))
        else:
            raise ValueError

    # deal with grad clip
    if cfg_.get('grad_clip'):
        if isinstance(cfg_.get('grad_clip'), float):
            cfg_['grad_clip'] = cfg_.get('grad_clip').get('value')
        elif 'global' in cfg_.get('grad_clip').get('name').lower():
            cfg_['grad_clip'] = paddle.nn.ClipGradByGlobalNorm(
                cfg_.get('grad_clip').get('value'))
        else:
            raise ValueError
    return OPTIMIZERS.get(name)(lr_scheduler, parameters=parameters, **cfg_)
