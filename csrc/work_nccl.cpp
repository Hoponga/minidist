#include "work_nccl.h"

std::shared_ptr<WorkNCCL> WorkNCCL::record(
    cudaStream_t stream,
    std::vector<torch::Tensor> tensors) {
}

bool WorkNCCL::is_completed() {
}

void WorkNCCL::wait() {
}
