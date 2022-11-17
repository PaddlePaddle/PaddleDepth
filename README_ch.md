[comment]: <> ([English]&#40;README.md&#41; | 简体中文)

[comment]: <> (<p align="center">)

[comment]: <> ( <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>)

[comment]: <> (<p align="center">)

[comment]: <> (<p align="left">)

[comment]: <> (    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>)

[comment]: <> (    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>)

[comment]: <> (    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>)

[comment]: <> (    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>)

[comment]: <> (    <a href=""><img src="https://img.shields.io/pypi/format/PaddleOCR?color=c77"></a>)

[comment]: <> (    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>)

[comment]: <> (    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>)

[comment]: <> (</p>)

#PaddleDepth


## 简介

PaddleDepth旨在打造一套产业级的深度信息增强方案，助力开发者更低成本的搜集深度信息，
目前共包含深度图超分辨，深度图补全，单目深度估计及双目深度估计这四个子方向



[comment]: <> (## 📣 近期更新)

[comment]: <> (- **💥 直播预告：10.24-10.26日每晚8点半**，PaddleOCR研发团队详解PP-StructureV2优化策略。微信扫描下方二维码，关注公众号并填写问卷后进入官方交流群，获取直播链接与20G重磅OCR学习大礼包（内含PDF转Word应用程序、10种垂类模型、《动手学OCR》电子书等）)

[comment]: <> (<div align="center">)

[comment]: <> (<img src="https://user-images.githubusercontent.com/50011306/196944258-0eb82df1-d730-4b96-a350-c1d370fdc2b1.jpg"  width = "150" height = "150" />)

[comment]: <> (</div>)

[comment]: <> (- **🔥2022.8.24 发布 PaddleOCR [release/2.6]&#40;https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6&#41;**)

[comment]: <> (  - 发布[PP-StructureV2]&#40;./ppstructure/README_ch.md&#41;，系统功能性能全面升级，适配中文场景，新增支持[版面复原]&#40;./ppstructure/recovery/README_ch.md&#41;，支持**一行命令完成PDF转Word**；)

[comment]: <> (  - [版面分析]&#40;./ppstructure/layout/README_ch.md&#41;模型优化：模型存储减少95%，速度提升11倍，平均CPU耗时仅需41ms；)

[comment]: <> (  - [表格识别]&#40;./ppstructure/table/README_ch.md&#41;模型优化：设计3大优化策略，预测耗时不变情况下，模型精度提升6%；)

[comment]: <> (  - [关键信息抽取]&#40;./ppstructure/kie/README_ch.md&#41;模型优化：设计视觉无关模型结构，语义实体识别精度提升2.8%，关系抽取精度提升9.1%。)

[comment]: <> (- **🔥2022.8 发布 [OCR场景应用集合]&#40;./applications&#41;**)

[comment]: <> (  - 包含数码管、液晶屏、车牌、高精度SVTR模型、手写体识别等**9个垂类模型**，覆盖通用，制造、金融、交通行业的主要OCR垂类应用。)


[comment]: <> (- **2022.8 新增实现[8种前沿算法]&#40;doc/doc_ch/algorithm_overview.md&#41;**)

[comment]: <> (  - 文本检测：[FCENet]&#40;doc/doc_ch/algorithm_det_fcenet.md&#41;, [DB++]&#40;doc/doc_ch/algorithm_det_db.md&#41;)

[comment]: <> (  - 文本识别：[ViTSTR]&#40;doc/doc_ch/algorithm_rec_vitstr.md&#41;, [ABINet]&#40;doc/doc_ch/algorithm_rec_abinet.md&#41;, [VisionLAN]&#40;doc/doc_ch/algorithm_rec_visionlan.md&#41;, [SPIN]&#40;doc/doc_ch/algorithm_rec_spin.md&#41;, [RobustScanner]&#40;doc/doc_ch/algorithm_rec_robustscanner.md&#41;)

[comment]: <> (  - 表格识别：[TableMaster]&#40;doc/doc_ch/algorithm_table_master.md&#41;)


[comment]: <> (- **2022.5.9 发布 PaddleOCR [release/2.5]&#40;https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5&#41;**)

[comment]: <> (    - 发布[PP-OCRv3]&#40;./doc/doc_ch/ppocr_introduction.md#pp-ocrv3&#41;，速度可比情况下，中文场景效果相比于PP-OCRv2再提升5%，英文场景提升11%，80语种多语言模型平均识别准确率提升5%以上；)

[comment]: <> (    - 发布半自动标注工具[PPOCRLabelv2]&#40;./PPOCRLabel&#41;：新增表格文字图像、图像关键信息抽取任务和不规则文字图像的标注功能；)

[comment]: <> (    - 发布OCR产业落地工具集：打通22种训练部署软硬件环境与方式，覆盖企业90%的训练部署环境需求；)

[comment]: <> (    - 发布交互式OCR开源电子书[《动手学OCR》]&#40;./doc/doc_ch/ocr_book.md&#41;，覆盖OCR全栈技术的前沿理论与代码实践，并配套教学视频。)

[comment]: <> (> [更多]&#40;./doc/doc_ch/update.md&#41;)

## 🌟 特性

- **模型丰富**: 包含**深度图超分辨**、**深度补全**、**单目深度估计**、****双目深度估计****等**10+前沿算法，及4+自研模型**。
- **使用简洁**：模块化设计，解耦各个网络组件，开发者轻松搭建，快速得到高性能、定制化的算法。
- **公平对比**: 基于飞桨框架，在相同的训练策略和环境下公平比较了深度信息增强领域里面SOTA(state-of-the-art)的算法

<div align="center">
    <img src="\docs\images\ppdepth.png" width="600" />
</div>

## ⚡ 快速开始
> 点击下述超链接查看各个细分深度信息增强算法的使用方法
- [深度图超分辨](./Depth_super_resolution/README_cn.md)
- 深度补全
- 单目深度估计
- 双目深度估计

<a name="电子书"></a>

[comment]: <> (## 📚《动手学OCR》电子书)

[comment]: <> (- [《动手学OCR》电子书]&#40;./doc/doc_ch/ocr_book.md&#41;)


[comment]: <> (<a name="开源社区"></a>)

[comment]: <> (## 👫 开源社区)

[comment]: <> (- **📑项目合作：** 如果您是企业开发者且有明确的OCR垂类应用需求，填写[问卷]&#40;https://paddle.wjx.cn/vj/QwF7GKw.aspx&#41;后可免费与官方团队展开不同层次的合作。)

[comment]: <> (- **👫加入社区：** 微信扫描二维码并填写问卷之后，加入交流群领取20G重磅OCR学习大礼包)

[comment]: <> (  - **包括《动手学OCR》电子书** ，配套讲解视频和notebook项目；PaddleOCR历次发版直播课视频；)

[comment]: <> (  - **OCR场景应用模型集合：** 包含数码管、液晶屏、车牌、高精度SVTR模型、手写体识别等垂类模型，覆盖通用，制造、金融、交通行业的主要OCR垂类应用。)

[comment]: <> (  - PDF2Word应用程序；OCR社区优秀开发者项目分享视频。)

[comment]: <> (- **🏅️社区项目**：[社区项目]&#40;./doc/doc_ch/thirdparty.md&#41;文档中包含了社区用户**使用PaddleOCR开发的各种工具、应用**以及**为PaddleOCR贡献的功能、优化的文档与代码**等，是官方为社区开发者打造的荣誉墙，也是帮助优质项目宣传的广播站。  )

[comment]: <> (- **🎁社区常规赛**：社区常规赛是面向OCR开发者的积分赛事，覆盖文档、代码、模型和应用四大类型，以季度为单位评选并发放奖励，赛题详情与报名方法可参考[链接]&#40;https://github.com/PaddlePaddle/PaddleOCR/issues/4982&#41;。)

[comment]: <> (<div align="center">)

[comment]: <> (<img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/joinus.PNG"  width = "150" height = "150" />)

[comment]: <> (</div>)



[comment]: <> (<a name="模型下载"></a>)

[comment]: <> (## 🛠️ PP-OCR系列模型列表（更新中）)

[comment]: <> (| 模型简介                              | 模型名称                | 推荐场景        | 检测模型                                                     | 方向分类器                                                   | 识别模型                                                     |)

[comment]: <> (| ------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |)

[comment]: <> (| 中英文超轻量PP-OCRv3模型（16.2M）     | ch_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar&#41; / [训练模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar&#41; | [推理模型]&#40;https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar&#41; / [训练模型]&#40;https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar&#41; | [推理模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar&#41; / [训练模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar&#41; |)

[comment]: <> (| 英文超轻量PP-OCRv3模型（13.4M）     | en_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar&#41; / [训练模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar&#41; | [推理模型]&#40;https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar&#41; / [训练模型]&#40;https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar&#41; | [推理模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar&#41; / [训练模型]&#40;https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar&#41; |)

[comment]: <> (- 超轻量OCR系列更多模型下载（包括多语言），可以参考[PP-OCR系列模型下载]&#40;./doc/doc_ch/models_list.md&#41;，文档分析相关模型参考[PP-Structure系列模型下载]&#40;./ppstructure/docs/models_list.md&#41;)

[comment]: <> (### PaddleOCR场景应用模型)

[comment]: <> (| 行业 | 类别         | 亮点                               | 文档说明                                                     | 模型下载                                      |)

[comment]: <> (| ---- | ------------ | ---------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |)

[comment]: <> (| 制造 | 数码管识别   | 数码管数据合成、漏识别调优         | [光功率计数码管字符识别]&#40;./applications/光功率计数码管字符识别/光功率计数码管字符识别.md&#41; | [下载链接]&#40;./applications/README.md#模型下载&#41; |)

[comment]: <> (| 金融 | 通用表单识别 | 多模态通用表单结构化提取           | [多模态表单识别]&#40;./applications/多模态表单识别.md&#41;           | [下载链接]&#40;./applications/README.md#模型下载&#41; |)

[comment]: <> (| 交通 | 车牌识别     | 多角度图像处理、轻量模型、端侧部署 | [轻量级车牌识别]&#40;./applications/轻量级车牌识别.md&#41;           | [下载链接]&#40;./applications/README.md#模型下载&#41; |)

[comment]: <> (- 更多制造、金融、交通行业的主要OCR垂类应用模型（如电表、液晶屏、高精度SVTR模型等），可参考[场景应用模型下载]&#40;./applications&#41;)

[comment]: <> (<a name="文档教程"></a>)

[comment]: <> (## 📖 文档教程)

[comment]: <> (- [运行环境准备]&#40;./doc/doc_ch/environment.md&#41;)

[comment]: <> (- [PP-OCR文本检测识别🔥]&#40;./doc/doc_ch/ppocr_introduction.md&#41;)

[comment]: <> (    - [快速开始]&#40;./doc/doc_ch/quickstart.md&#41;)

[comment]: <> (    - [模型库]&#40;./doc/doc_ch/models_list.md&#41;)

[comment]: <> (    - [模型训练]&#40;./doc/doc_ch/training.md&#41;)

[comment]: <> (        - [文本检测]&#40;./doc/doc_ch/detection.md&#41;)

[comment]: <> (        - [文本识别]&#40;./doc/doc_ch/recognition.md&#41;)

[comment]: <> (        - [文本方向分类器]&#40;./doc/doc_ch/angle_class.md&#41;)

[comment]: <> (    - 模型压缩)

[comment]: <> (        - [模型量化]&#40;./deploy/slim/quantization/README.md&#41;)

[comment]: <> (        - [模型裁剪]&#40;./deploy/slim/prune/README.md&#41;)

[comment]: <> (        - [知识蒸馏]&#40;./doc/doc_ch/knowledge_distillation.md&#41;)

[comment]: <> (    - [推理部署]&#40;./deploy/README_ch.md&#41;)

[comment]: <> (        - [基于Python预测引擎推理]&#40;./doc/doc_ch/inference_ppocr.md&#41;)

[comment]: <> (        - [基于C++预测引擎推理]&#40;./deploy/cpp_infer/readme_ch.md&#41;)

[comment]: <> (        - [服务化部署]&#40;./deploy/pdserving/README_CN.md&#41;)

[comment]: <> (        - [端侧部署]&#40;./deploy/lite/readme.md&#41;)

[comment]: <> (        - [Paddle2ONNX模型转化与预测]&#40;./deploy/paddle2onnx/readme.md&#41;)

[comment]: <> (        - [云上飞桨部署工具]&#40;./deploy/paddlecloud/README.md&#41;)

[comment]: <> (        - [Benchmark]&#40;./doc/doc_ch/benchmark.md&#41;)

[comment]: <> (- [PP-Structure文档分析🔥]&#40;./ppstructure/README_ch.md&#41;)

[comment]: <> (    - [快速开始]&#40;./ppstructure/docs/quickstart.md&#41;)

[comment]: <> (    - [模型库]&#40;./ppstructure/docs/models_list.md&#41;)

[comment]: <> (    - [模型训练]&#40;./doc/doc_ch/training.md&#41;)

[comment]: <> (        - [版面分析]&#40;./ppstructure/layout/README_ch.md&#41;)

[comment]: <> (        - [表格识别]&#40;./ppstructure/table/README_ch.md&#41;)

[comment]: <> (        - [关键信息提取]&#40;./ppstructure/kie/README_ch.md&#41;)

[comment]: <> (    - [推理部署]&#40;./deploy/README_ch.md&#41;)

[comment]: <> (        - [基于Python预测引擎推理]&#40;./ppstructure/docs/inference.md&#41;)

[comment]: <> (        - [基于C++预测引擎推理]&#40;./deploy/cpp_infer/readme_ch.md&#41;)

[comment]: <> (        - [服务化部署]&#40;./deploy/hubserving/readme.md&#41;)

[comment]: <> (- [前沿算法与模型🚀]&#40;./doc/doc_ch/algorithm_overview.md&#41;)

[comment]: <> (    - [文本检测算法]&#40;./doc/doc_ch/algorithm_overview.md&#41;)

[comment]: <> (    - [文本识别算法]&#40;./doc/doc_ch/algorithm_overview.md&#41;)

[comment]: <> (    - [端到端OCR算法]&#40;./doc/doc_ch/algorithm_overview.md&#41;)

[comment]: <> (    - [表格识别算法]&#40;./doc/doc_ch/algorithm_overview.md&#41;)

[comment]: <> (    - [关键信息抽取算法]&#40;./doc/doc_ch/algorithm_overview.md&#41;)

[comment]: <> (    - [使用PaddleOCR架构添加新算法]&#40;./doc/doc_ch/add_new_algorithm.md&#41;)

[comment]: <> (- [场景应用]&#40;./applications&#41;)

[comment]: <> (- 数据标注与合成)

[comment]: <> (    - [半自动标注工具PPOCRLabel]&#40;./PPOCRLabel/README_ch.md&#41;)

[comment]: <> (    - [数据合成工具Style-Text]&#40;./StyleText/README_ch.md&#41;)

[comment]: <> (    - [其它数据标注工具]&#40;./doc/doc_ch/data_annotation.md&#41;)

[comment]: <> (    - [其它数据合成工具]&#40;./doc/doc_ch/data_synthesis.md&#41;)

[comment]: <> (- 数据集)

[comment]: <> (    - [通用中英文OCR数据集]&#40;doc/doc_ch/dataset/datasets.md&#41;)

[comment]: <> (    - [手写中文OCR数据集]&#40;doc/doc_ch/dataset/handwritten_datasets.md&#41;)

[comment]: <> (    - [垂类多语言OCR数据集]&#40;doc/doc_ch/dataset/vertical_and_multilingual_datasets.md&#41;)

[comment]: <> (    - [版面分析数据集]&#40;doc/doc_ch/dataset/layout_datasets.md&#41;)

[comment]: <> (    - [表格识别数据集]&#40;doc/doc_ch/dataset/table_datasets.md&#41;)

[comment]: <> (    - [关键信息提取数据集]&#40;doc/doc_ch/dataset/kie_datasets.md&#41;)

[comment]: <> (- [代码组织结构]&#40;./doc/doc_ch/tree.md&#41;)

[comment]: <> (- [效果展示]&#40;#效果展示&#41;)

[comment]: <> (- [《动手学OCR》电子书📚]&#40;./doc/doc_ch/ocr_book.md&#41;)

[comment]: <> (- [开源社区]&#40;#开源社区&#41;)

[comment]: <> (- FAQ)

[comment]: <> (    - [通用问题]&#40;./doc/doc_ch/FAQ.md&#41;)

[comment]: <> (    - [PaddleOCR实战问题]&#40;./doc/doc_ch/FAQ.md&#41;)

[comment]: <> (- [参考文献]&#40;./doc/doc_ch/reference.md&#41;)

[comment]: <> (- [许可证书]&#40;#许可证书&#41;)


<a name="效果展示"></a>

## 👀 效果展示


###深度补全


<div align="center">
    <img src="\docs\images\completion.gif" width="600" />
</div>


###单目深度估计

###双目深度估计


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