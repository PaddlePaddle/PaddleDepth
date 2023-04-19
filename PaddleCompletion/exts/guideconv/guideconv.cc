#include <vector>

#include <paddle/extension.h>


void Conv2d_LF_Cuda(paddle::Tensor x, paddle::Tensor y, paddle::Tensor z, size_t N1, size_t N2, size_t Ci, size_t Co, size_t B,
                    size_t K);

void
Conv2d_LB_Cuda(paddle::Tensor x, paddle::Tensor y, paddle::Tensor gx, paddle::Tensor gy, paddle::Tensor gz, size_t N1, size_t N2, size_t Ci,
                    size_t Co, size_t B, size_t K);


std::vector<paddle::Tensor> Conv2dLocal_F(
        const paddle::Tensor& a, // BCHW
        const paddle::Tensor& b // BCKKHW
) {
    int N1, N2, Ci, Co, K, B;
    B = a.shape()[0];
    Ci = a.shape()[1];
    N1 = a.shape()[2];
    N2 = a.shape()[3];
    Co = Ci;
    K = sqrt(b.shape()[1] / Co);
    auto c = paddle::experimental::zeros_like(a);
    Conv2d_LF_Cuda(a, b, c, N1, N2, Ci, Co, B, K);
    return {c};
}


std::vector<paddle::Tensor> Conv2dLocal_B(
        const paddle::Tensor& a,
        const paddle::Tensor& b,
        const paddle::Tensor& gc
) {
    int N1, N2, Ci, Co, K, B;
    B = a.shape()[0];
    Ci = a.shape()[1];
    N1 = a.shape()[2];
    N2 = a.shape()[3];
    Co = Ci;
    K = sqrt(b.shape()[1] / Co);
    auto ga = paddle::experimental::zeros_like(a);
    auto gb = paddle::experimental::zeros_like(b);
    Conv2d_LB_Cuda(a, b, ga, gb, gc, N1, N2, Ci, Co, B, K);
    return {ga, gb};
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> a_shape,
                                             std::vector<int64_t> b_shape) {
  return {a_shape};
}

std::vector<std::vector<int64_t>>
InferBackShape(std::vector<int64_t> a_shape,
               std::vector<int64_t> b_shape) {
  return {a_shape, b_shape};
}

std::vector<paddle::DataType> InferDtype(paddle::DataType a_dtype,
                                         paddle::DataType b_dtype) {
  return {a_dtype};
}

PD_BUILD_OP(GuideConv)
    .Inputs({"a", "b"})
    .Outputs({"c"})
    .SetKernelFn(PD_KERNEL(Conv2dLocal_F))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));

PD_BUILD_GRAD_OP(GuideConv)
    .Inputs({"a", "b", paddle::Grad("c")})
    .Outputs({paddle::Grad("a"), paddle::Grad("b")})
    .SetKernelFn(PD_KERNEL(Conv2dLocal_B))
    .SetInferShapeFn(PD_INFER_SHAPE(InferBackShape));
