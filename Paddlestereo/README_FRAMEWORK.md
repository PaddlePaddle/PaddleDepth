# 数据集搭建

模板源代码：[https://github.com/Archaic-Atom/Template-jf](https://github.com/Archaic-Atom/Template-jf)

模板结构详情见002模板基本结构

## 1数据接口模板简介

相关文件位置：Source/UserModelImplementation/Dataloaders/your_dataloader.py

该模块主要是：传递数据的读取、保存、以及显示方式给框架。

模板如下：

```Python
# -*- coding: utf-8 -*-
import time
import JackFramework as jf
# import UserModelImplementation.user_define as user_def


class YourDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__imgs_num = 0
        self.__start_time = 0

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        # args = self.__args
        # return dataset
        return None

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        # args = self.__args
        # return dataset
        return None

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [], []
            # return input_data, supplement
        return [], []

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        # args = self.__args
        # save method
        pass

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])

```

## 2函数说明

### 2.1 初始化方法

```Python
def __init__(self, args: object) -> object:
    super().__init__(args)

```

- args：相关参数，包括框架中自带的参数和用户自定义的参数。用户自定义的方法见003 超参数设置

**Example**：

```Python
def __init__(self, args: object) -> object:
    super().__init__(args)
    self.__args = args
    self.__result_str = jf.ResultStr() # covert the list of result to string.
    self.__train_dataset = None
    self.__val_dataset = None
    self.__imgs_num = 0
    self.__start_time = 0
```

### 2.2 训练数据集或测试数据集

```Python
    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        # args = self.__args
        # return dataset
        return None
```

- path：传递框架自带参数中的trainListPath。自带参数列表见006 载入框架；
- is_training：是否是训练阶段；

注意：当训练阶段，需要返回训练相关的数据集，而测试阶段，需要返回测试的数据集。

**Example**：

```Python
    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        # return dataset
        self.__train_dataset = torchvision.datasets.MNIST(
            'MNIST', train=True, download=True, transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((user_def.MIST_MEAN,), (user_def.MIST_VAR,))
            ]))
        return self.__train_dataset
```

注意：本例中仅用于训练，因此无测试数据集。

### 2.3 验证数据集

```Python
    def get_val_dataset(self, path: str) -> object:
        # return dataset
        # args = self.__args
        # return dataset
        return None
```

- path：传递框架自带参数中的valListPath。自带参数列表见006 载入框架；

**Example**：

```Python
    def get_val_dataset(self, path: str) -> object:
        # return dataset
        args = self.__args
        # return dataset
        self.__val_dataset = torchvision.datasets.MNIST(
            'MNIST', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (user_def.MIST_MEAN,), (user_def.MIST_VAR,))
            ]))
        return self.__val_dataset
```

### 2.4 数据划分方法

```Python
    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [], []
            # return input_data, supplement
        return [], []
```

- batch_data：一个批次的数据；
- is_training：是否是训练阶段；
    - 当训练阶段时候，需要返回两个list，一个是输入数据，一个是标签数据；
    - 当测试阶段时候，需要返回两个list，一个是输入数据，一个是额外的数据（例如，图像的名字，裁剪的尺寸等）；

**Example**：

```Python
    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [batch_data[0]], [batch_data[1]]
            # return input_data, supplement
        return [batch_data[0]], [batch_data[1]]

```

```Python
    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [batch_data[ID_IMG_L], batch_data[ID_IMG_R]], [batch_data[ID_DISP]]
            # return input_data, supplement
        return [batch_data[ID_IMG_L], batch_data[ID_IMG_R]], \
            [batch_data[ID_TOP_PAD], batch_data[ID_LEFT_PAD], batch_data[ID_NAME]]

```

### 2.5 记录训练过程

```Python
    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

```

- epoch：当前代数；
- loss：当前代数的平均loss的结果；
- acc：当前代数的平均acc的结果；

代码解析：该部分针对的是1个模型，因此是loss[0]和acc[0]的结果；若有多个模型，请使用loss[id]和acc[id]得到对应id的损失函数和精度信息。

```Python
self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
```

会将对应的内容转换为str类型；例如：

```Python
>>> result_str = jf.ResultStr()
>>> epoch = 45
>>> loss = [0.6, 0.7, 0.8]
>>> acc = [0.98, 0.99 ,1.0]
>>> duration = 1000
>>> result_str.training_result_str(epoch, loss,acc, duration,True)
'[TrainProcess] e: 45, l0: 0.600000, l1: 0.700000, l2: 0.800000, a0: 0.980000, a1: 0.990000, a2: 1.000000 (1000.000 s/epoch)'
>>> result_str.training_result_str(epoch, loss,acc, duration, False)
'[ValProcess] e: 45, l0: 0.600000, l1: 0.700000, l2: 0.800000, a0: 0.980000, a1: 0.990000, a2: 1.000000 (1000.000 s/epoch)'
```

然后，该函数会在对应的文件中写下日志，如下所示。

```Python
jf.log.info(info_str)
```

### 2.6 记录验证过程

```Python
    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

```

- epoch：当前代数；
- loss：当前代数的平均loss的结果；
- acc：当前代数的平均acc的结果；

### 2.7 数据保存过程

```Python
    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        # args = self.__args
        # save method
        pass
```

- output_data：输出的数据；该部分是用户在model类中交给框架的数据；详情见005 模型搭建（Model）
- supplement：支撑材料；该部分是用户在dataloader中交给框架的数据；详情见2.4 数据划分方法
- img_id：图像的id；
- model_id：模型编号；

**Example**：

```Python
    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args = self.__args
        off_set = 1
        last_position = len(output_data) - off_set
        # last_position = 0

        if model_id == 0:
            self.__saver.save_output(output_data[last_position].cpu().detach().numpy(),
                                     img_id, args.dataset, supplement,
                                     time.time() - self.__start_time)
```

### 2.8 进度条显示内容

```Python
    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])
```

- epoch：当前代数；
- loss：所有模型的loss损失函数。
- acc：所有模型的acc精度。

注意：本例中仅仅显示第一个模型的损失和精度，如果涉及多模型，请重新组合数据。

# 模型搭建

模板源代码：[https://github.com/Archaic-Atom/Template-jf](https://github.com/Archaic-Atom/Template-jf)

模板结构详情见002模板基本结构

## 1模型接口模板简介

相关文件位置：Source/UserModelImplementation/Models/Your_Model/inference.py

该模块主要是：传递模型的结构、计算loss方式、以及计算acc的方式等给框架。

模板如下：

```Python
# -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import JackFramework as jf
# import UserModelImplementation.user_define as user_def


class YourModelInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        # args = self.__args
        # return model
        return []

    def optimizer(self, model: list, lr: float) -> list:
        # args = self.__args
        # return opt and sch
        return [], []

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        pass

    def inference(self, model: object, input_data: list, model_id: int) -> list:
        # args = self.__args
        # return output
        return []

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        return []

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        return []

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None

```

## 2函数说明

### 2.1初始化函数

初始化函数用于初始化整个类。

```Python
    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
```

- args：相关参数，包括框架中自带的参数和用户自定义的参数。用户自定义的方法见003 超参数设置。

**Example：**

```Python
    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__lr = args.lr

```

### 2.2传递模型

传递模型给框架，返回值为一个list（可以优化多个模型，但是需要相应数量的优化器）。

```Python
    def get_model(self) -> list:
        # args = self.__args
        # return model
        return []
```

**Example：**

```Python
from .model import ConvNet

    def get_model(self) -> list:
        args = self.__args
        model = ConvNet(user_def.CHANNELS_NUM)
        # return model
        return [model]

```

### 2.3传递优化器和学习策略

传递优化器和学习策略，返回值为两个list

```Python
    def optimizer(self, model: list, lr: float) -> list:
        # args = self.__args
        # return opt and sch
        return [], []
```

- model：模型列表。有用户在在get_model函数中提供。详情见2.2传递模型；
- lr：学习率。由用户在框架中设置学习率提供，参数列表见006 载入框架。

**Example：**

```Python
    @staticmethod
    def lr_lambda(epoch: int) -> float:
        max_warm_up_epoch = 10
        convert_epoch = 50
        off_set = 1
        lr_factor = 1.0

        factor = ((epoch + off_set) / max_warm_up_epoch) if epoch < max_warm_up_epoch \
            else lr_factor if (epoch >= max_warm_up_epoch and epoch < convert_epoch) \
            else lr_factor * 0.25
        return factor

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        # return opt
        # opt = optim.AdamW(model[0].parameters(), lr=lr, weight_decay=1e-4)
        opt = optim.Adam(model[0].parameters(), lr=lr)

        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=CRNetInterface.lr_lambda)
        else:
            sch = None

        return [opt], [sch]

```

注意，本例中只有一个模型。

### 2.4学习策略的使用方式

告诉框架如何使用用户的学习策略；

```Python
    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        pass
```

- sch：学习策略的对象；由用户在optimizer函数传递，详情见2.3传递优化器和学习策略。
- ave_loss：为一代的平均损失函数；由用户在loss函数计算得到每一步的损失，框架自动计算每一代平均值。
- sch_id：sch的编号，由用户在optimizer函数传递的sch的长度决定。

**Example：**

```Python
    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        if self.MODEL_ID == sch_id:
            sch.step()

```

### 2.5模型的使用方式

模型的使用方式

```Python
    def inference(self, model: object, input_data: list, model_id: int) -> list:
        # args = self.__args
        # return output
        return []
```

- model：模型列表；为用户在get_model中提供，详情见2.2传递模型；
- input_data：输入数据；由用户在dataloader构建中split_data提供的划分方式决定，详情见004 数据集搭建（Dataloader）中的2.4 数据划分方法；
- model_id：当前处理的模型编号。由用户在get_model函数传递的model的长度决定。

**Example：**

```Python
    def inference(self, model: object, input_data: list, model_id: int) -> list:
        disp_1 = None
        disp_2 = None
        disp_3 = None
        if self.MODEL_ID == model_id:
            disp_1, disp_2, disp_3 = model(input_data[0], input_data[1])

        return [disp_1, disp_2, disp_3]
```

### 2.6损失函数的使用方式

计算每一个批次的损失函数

```Python
    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        return []
```

- output_data：输出数据；由用户在inference接口中返回，详情见2.5模型的使用方式
- label_data：标签数据；由用户在dataloader构建中split_data提供的划分方式决定，详情见004 数据集搭建（Dataloader）中的2.4 数据划分方法；
- model_id：当前处理的模型编号。由用户在get_model函数传递的model的长度决定。

**Example：**

```Python
    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        total_loss = None
        loss_0 = None
        loss_1 = None
        loss_2 = None

        if self.MODEL_ID == model_id:
            args = self.__args
            loss_0 = jf.Loss.smooth_l1(output_data[0], label_data[0],
                                       args.startDisp, args.startDisp + args.dispNum)
            loss_1 = jf.Loss.smooth_l1(output_data[1], label_data[0],
                                       args.startDisp, args.startDisp + args.dispNum)
            loss_2 = jf.Loss.smooth_l1(output_data[2], label_data[0],
                                       args.startDisp, args.startDisp + args.dispNum)
            total_loss = loss_0 + loss_1 + loss_2

        return [total_loss, loss_0, loss_1, loss_2]
```

### 2.7精度的计算方式

计算每一个批次的精度的方式

```Python
    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        return []
```

- output_data：输出数据；由用户在inference接口中返回，详情见2.5模型的使用方式；
- label_data：标签数据；由用户在dataloader构建中split_data提供的划分方式决定，详情见004 数据集搭建（Dataloader）中的2.4 数据划分方法；
- model_id：当前处理的模型编号。由用户在get_model函数传递的model的长度决定。

**Example：**

```Python
    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        acc_0 = None
        mae_0 = None
        acc_1 = None
        mae_1 = None
        acc_2 = None
        mae_2 = None

        if self.MODEL_ID == model_id:
            acc_0, mae_0 = jf.SMAccuracy.d_1(output_data[0], label_data[0])
            acc_1, mae_1 = jf.SMAccuracy.d_1(output_data[1], label_data[0])
            acc_2, mae_2 = jf.SMAccuracy.d_1(output_data[2], label_data[0])

        return [acc_0[1], acc_1[1], acc_2[1], mae_0, mae_1, mae_2]
```

### 2.8预处理的方式（可选）

每一代预处理方式

```Python
    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

```

- epoch：当前代数；
- rank：当前显卡编号；（由于DDP是多进程工作，所以将显卡编号返回）。

### 2.9后处理的方式（可选）

每一代后处理的方式

```Python
    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass


```

- epoch：当前代数；
- rank：当前显卡编号；（由于DDP是多进程工作，所以将显卡编号返回）
- ave_tower_loss：所有模型的平均损失；双重列表，通过ave_tower_loss[model_id]可以得到对应模型的平均损失；
- ave_tower_acc：所有模型的平均精度；双重列表，通过ave_tower_acc[model_id]可以得到对应模型的平均精度；

### 2.10加载模型的方式（可选）

可以选择加载模型参数的方式（如果需要使用用户自己加载方式，请返回True；否则将按框架自己的方式加载模型）

```Python
    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

```

- model：模型对象；由用户在get_model中提供，详情见2.2传递模型；
- checkpoint：权重字典；由用户在框架自带参数modelDir目录下得到的权重，相关参数说明详情见006 载入框架；
- model_id：当前处理的模型编号。由用户在get_model函数传递的model的长度决定。

**Example：**

```Python
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        model.load_state_dict(checkpoint['model_0'], strict=True)
        jf.log.info("Model loaded successfully")
        return True
```

### 2.11加载优化器的方式（可选）

可以选择加载优化器参数的方式（如果需要使用用户自己加载方式，请返回True；否则将按框架自己的方式加载模型）

```Python
    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False


```

- opt：优化器对象；由用户在optimizer函数传递，详情见2.3传递优化器和学习策略。
- checkpoint：权重字典；由用户在框架自带参数modelDir目录下得到的权重，相关参数说明详情见006 载入框架；
- opt：当前处理的优化器编号。由用户在optimizer函数传递的opt的长度决定。

**Example：**

```Python
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        opt.load_state_dict(checkpoint['optimizer'])
        jf.log.info("Model loaded successfully")
        return True
```

### 2.12保存模型的方式（可选）

可以选择保存模型的字典(如果需要按用户要求的方式进行保存，请返回相应需要保存的字典；否则，返回None，将按框架自己的方式保存模型)

```Python
    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
```

- epoch：当前代数；
- model_list：模型的列表；由用户在get_model中提供，详情见2.2传递模型；
- opt_list：优化器列表；由用户在optimizer函数传递，详情见2.3传递优化器和学习策略。

# 载入框架

模板源代码：[https://github.com/Archaic-Atom/Template-jf](https://github.com/Archaic-Atom/Template-jf)

模板结构详情见002模板基本结构

## 1数据接口导出

目的：将设置好数据接口与外部连通；

文件所在位置: Source/UserModelImplementation/Dataloaders/**init**.py

```Python
# -*- coding: utf-8 -*-
import JackFramework as jf

from .your_dataloader import YourDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('YourDataloader'):
            jf.log.info("Enter the your dataloader")
            dataloader = YourDataloader(args)
            break
        if case(''):
            dataloader = None
            jf.log.error("The dataloader's name is error!!!")
    return dataloader

```

**Example：**

```Python
# -*- coding: utf-8 -*
import JackFramework as jf

from .mnist_dataloader import MNISTDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('mnist'):
            jf.log.info("Enter the mnist dataloader")
            dataloader = MNISTDataloader(args)
            break
        if case(''):
            dataloader = None
            jf.log.error("The dataloader's name is error!!!")
    return dataloader

```

## 2模型接口导出

目的：将设置好模板接口与外部连通；

文件所在位置: Source/UserModelImplementation/Models/**init**.py

```Python
# -*- coding: utf-8 -*
import JackFramework as jf

from .Your_Model.inference import YourModelInterface


def model_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('YourMode'):
            jf.log.info("Enter the YourMode model")
            model = YourModelInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model
```

**Example：**

```Python
# -*- coding: utf-8 -*
import JackFramework as jf

from .ConvNet.inference import ConvNetInterface


def models_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('ConvNet'):
            jf.log.info("Enter the ConvNet model")
            model = ConvNetInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model

```

## 3. 设置启动参数

在Scripts/start_*.sh中设置启动脚本，如下所示；

```Vim script
#!/bin/bash
# parameters
tensorboard_port=6235
dist_port=8801
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u Source/main.py \
                        --batchSize 256\
                        --gpu 8 \
                        --trainListPath ./Datasets/mnist_dataset.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 24 \
                        --maxEpochs 200 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist True \
                        --modelName ConvNet \
                        --port ${dist_port} \
                        --dataset mnist > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"
echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ./log --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f TrainRun.log
```

**框架支持的参数如下：**

|Args|Type|Description|Default|
|-|-|-|-|
|mode|[str]|train or test|train|
|gpu|[int]|the number of gpus|2|
|auto_save_num|[int]|the number of interval save|1|
|dataloaderNum|[int]|the number of dataloders|8|
|pretrain|[bool]|is a new traning process|False|
|ip|[str]|used for distributed training|127.0.0.1|
|port|[str]|used for distributed training|8086|
|dist|[bool]|distributed training (DDP)|True|
|trainListPath|[str]|the list for training or testing|./Datasets/*.csv|
|valListPath|[str]|the list for validate process|./Datasets/*.csv|
|outputDir|[str]|the folder for log file|./Result/|
|modelDir|[str]|the folder for saving model|./Checkpoint/|
|resultImgDir|[str]|the folder for output|./ResultImg/|
|log|[str]|the folder for tensorboard|./log/|
|sampleNum|[int]|the number of sample for data|1|
|batchSize|[int]|batch size|4|
|lr|[float]|leanring rate|0.001|
|maxEpochs|[int]|training epoch|30|
|imgWidth|[int]|the croped width|512|
|imgHeight|[int]|the croped height|256|
|imgNum|[int]|the number of images for tranin|35354|
|valImgNum|[int]|the number of images for val|200|
|modelName|[str]|the model's name|NLCA-Net|
|dataset|[str]|the dataset's name|SceneFlow|


## 4启动训练或者测试程序

```Bash
(JackFramework-torch1.7.1)$ ./Scripts/start_train_mnist_conv_net.sh 
```
