#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

class WorkNCCL {
public:
    static std::shared_ptr<WorkNCCL> record(
        cudaStream_t stream,
        std::vector<torch::Tensor> tensors);

    bool is_completed();
    void wait();

private:
    int device_ = -1;
    cudaEvent_t done_ = nullptr;
    std::vector<torch::Tensor> tensors_;
};
