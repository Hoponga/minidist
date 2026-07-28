#include "process_group_nccl.h"

#include "error.h"

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::string& unique_id_bytes,
    int global_rank,
    int group_rank,
    std::vector<int> global_ranks,
    int device)
    : global_rank_(global_rank),
      group_rank_(group_rank),
      device_(device),
      global_ranks_(std::move(global_ranks)) {
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
}

int ProcessGroupNCCL::rank() const {
}

int ProcessGroupNCCL::size() const {
}

int ProcessGroupNCCL::to_group_rank(int global_rank) const {
}

std::shared_ptr<WorkNCCL> ProcessGroupNCCL::all_reduce(
    torch::Tensor tensor,
    ReduceOp op,
    bool async_op) {
}

std::shared_ptr<WorkNCCL> ProcessGroupNCCL::broadcast(
    torch::Tensor tensor,
    int global_src,
    bool async_op) {
}

std::shared_ptr<WorkNCCL> ProcessGroupNCCL::reduce(
    torch::Tensor tensor,
    int global_dst,
    ReduceOp op,
    bool async_op) {
}

std::shared_ptr<WorkNCCL> ProcessGroupNCCL::all_gather_into_tensor(
    torch::Tensor output,
    torch::Tensor input,
    bool async_op) {
}

std::shared_ptr<WorkNCCL> ProcessGroupNCCL::reduce_scatter_tensor(
    torch::Tensor output,
    torch::Tensor input,
    ReduceOp op,
    bool async_op) {
}

void ProcessGroupNCCL::barrier() {
}

void ProcessGroupNCCL::destroy() {
}
