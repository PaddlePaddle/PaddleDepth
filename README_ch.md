
# PaddleDepth: 飞桨深度信息增强开发套件

<div align="center">

[English](README.md)| 简体中文

</div>

PaddleDepth旨在打造一套产业级的深度信息增强方案，助力开发者更低成本的搜集深度信息，目前共包含深度图超分辨，深度图补全，单目深度估计及双目深度估计这四个子方向

下面是通过深度补全，单目深度估计及双目深度估计进行三维重建的结果演示

https://user-images.githubusercontent.com/57089550/202428765-121ec2d8-2ecc-4b2e-bd39-fdbb679eed58.mp4

## 🌟 特性

- **模型丰富**: 包含**深度图超分辨**、**深度补全**、**单目深度估计**、****双目深度估计****等10+前沿算法，及4+自研模型。
- **使用简洁**：模块化设计，解耦各个网络组件，开发者轻松搭建，快速得到高性能、定制化的算法。
- **公平对比**: 基于飞桨框架，在相同的训练策略和环境下公平比较了深度信息增强领域里面SOTA(state-of-the-art)的算法

<div align="center">
    <img src="https://user-images.githubusercontent.com/57089550/202442392-84e9ab8b-de9d-489d-b6a8-944661e30b01.png" width = "600" />
</div>

## ⚡ 快速开始
> 点击下述超链接查看各个细分深度信息增强算法的使用方法
- [深度图超分辨](./Depth_super_resolution/README_cn.md)
- [深度补全](./PaddleCompletion/README_zh-CN.md)
- [单目深度估计](./PaddleMono/README_zh-CN.md)
- [双目深度估计](./Paddlestereo/README_zh-CN.md)


<a name="效果展示"></a>

## 👀 效果展示


### 深度补全
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/docs/images/completion.gif" width = "400" />
</div>

### 单目深度估计
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/docs/images/monocular.gif" width = "400" />
</div>

### 双目深度估计
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/docs/images/stereo.gif" width = "400" />
</div>


## 贡献

PaddleDepth工具箱目前还在积极维护与完善过程中。 我们非常欢迎外部开发者为Paddle Depth提供新功能\模型。 如果您有这方面的意愿的话，请往我们的邮箱或者issue里面反馈
## 感谢
PaddleDepth是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。
我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 
我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 许可证书
本项目的发布受<LICENSE>Apache 2.0 license</a>许可认证。

## 联系方式

- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@baidu.com
