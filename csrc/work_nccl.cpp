#include "work_nccl.h"

std::shared_ptr<WorkNCCL> WorkNCCL::record(
    cudaStream_t stream,
    std::vector<torch::Tensor> tensors) {
        return nullptr; 
}

bool WorkNCCL::is_completed() {
    return false; 
}

void WorkNCCL::wait() {
    return; 
}
