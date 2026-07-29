#include "tensor_utils.h"

void validate_tensor(const torch::Tensor& tensor, int device) {
    TORCH_CHECK(tensor.is_cuda(), "tensor must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(tensor.get_device() == device, "tensor is on the wrong CUDA device");
}

ncclDataType_t to_nccl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return ncclFloat32;
        case at::kDouble: return ncclFloat64;
        case at::kHalf: return ncclFloat16;
        case at::kBFloat16: return ncclBfloat16;
        case at::kInt: return ncclInt32;
        case at::kLong: return ncclInt64;
        case at::kChar: return ncclInt8;
        case at::kByte: return ncclUint8;
        default:
            TORCH_CHECK(false, "unsupported dtype for NCCL");
    }
}

ncclRedOp_t to_nccl_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM: return ncclSum;
        case ReduceOp::PRODUCT: return ncclProd;
        case ReduceOp::MIN: return ncclMin;
        case ReduceOp::MAX: return ncclMax;
        case ReduceOp::AVG: return ncclAvg;
    }
    TORCH_CHECK(false, "unsupported reduce op");
}
