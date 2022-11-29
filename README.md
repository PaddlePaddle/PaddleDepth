# PaddleDepth

<div align="center">

[English](README.md)| [简体中文](README_ch.md)

</div>

PaddleDepth is a lightweight, easy-to-extend, easy-to-learn, high-performance, and for-fair-comparison toolkit based 
on PaddlePaddle for depth information argumentation. It aims to create a industrial-level depth information enhancement solution to help developers using a lower cost to collect depth information.


Currently, the proposed PaddleDepth repo contains four depth information argumentation methods, including **depth super resolution, depth completion, monocular depth estimation, and stereo depth estimation**. Below is the point cloud reconstruction result by depth completion, monocular depth estimation, and stereo depth estimation.

https://user-images.githubusercontent.com/57089550/202428765-121ec2d8-2ecc-4b2e-bd39-fdbb679eed58.mp4



## 🌟 Features

- **Rich model library**: Paddledepth provides over 10+ SOTA models including **depth super resolution, depth completion, monocular depth estimation, and stereo depth estimation**, in which 4+ models are **self-developed and first open sourced**.
- **Simple to use**: Modular design, decoupling each network component; easy for developers to build; quick access to high-performance, customized algorithm.
- **Fair comparison**: All the included algorithms are trained in the same environment and training strategy for fair comparison.

<div align="center">
    <img src="https://user-images.githubusercontent.com/57089550/202442392-84e9ab8b-de9d-489d-b6a8-944661e30b01.png" width = "600" />
</div>!

## ⚡ Quick Experience
> click the hyperlinks below to find the usage guidence for each task
- [depth super resolution](./Depth_super_resolution/README.md)
- depth completion
- [monocular depth estimation](./PaddleMono/README.md)
- stereo depth estimation


<a name="效果展示"></a>

## 👀 Visualization


### depth completion
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/docs/images/completion.gif" width = "400" />
</div>

### monocular depth estimation
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/docs/images/monocular.gif" width = "400" />
</div>

### stereo depth estimation (stereo matching)
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/docs/images/stereo.gif" width = "400" />
</div>



## Contribution

The toolkit is under active development and contributions are welcome!  Feel free to submit issues or emails to ask questions or contribute your code. 
If you would like to implement new features, please submit a issue or emails to discuss with us first.

## Acknowledgement
PaddleDepth is an open source project that is contributed by researchers and engineers 
from various colleges and companies. 
We appreciate all the contributors who implement their methods or add new features, 
as well as users who give valuable feedbacks. 
We wish that the toolbox and benchmark could serve the growing research community by 
providing a flexible toolkit to reimplement existing methods and develop their new algorithms.

## License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleDepth/blob/develop/LICENSE">Apache 2.0 license</a>

## Contact

- [Zhelun Shen](https://github.com/gallenszl): shenzhelun@baidu.com
