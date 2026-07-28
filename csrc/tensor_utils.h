#pragma once

#include <torch/extension.h>
#include <nccl.h>

enum class ReduceOp {
    SUM = 0,
    PRODUCT = 1,
    MIN = 2,
    MAX = 3,
    AVG = 4,
};

void validate_tensor(const torch::Tensor& tensor, int device);

ncclDataType_t to_nccl_dtype(at::ScalarType dtype);
ncclRedOp_t to_nccl_op(ReduceOp op);
