from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="ext_ops",
    ext_modules=CUDAExtension(sources=["guideconv/guideconv.cc", "guideconv/guideconv.cu"]),
)
