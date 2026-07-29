#include "process_group_nccl.h"

#include <ATen/cuda/CUDAContext.h>

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
        if (unique_id_bytes.size() != sizeof(ncclUniqueId)) {
            throw std::invalid_argument("unique_id_bytes must be of size " + std::to_string(sizeof(ncclUniqueId)) + " but instead is " + std::to_string(unique_id_bytes.size()));

        }

        ncclUniqueId unique_id; 
        std::memcpy(&unique_id, unique_id_bytes.data(), sizeof(unique_id)); 

        CUDACHECK(cudaSetDevice(device)); 
        NCCLCHECK(ncclCommInitRank(&comm_, global_ranks_.size(), unique_id, group_rank_)); 

}

ProcessGroupNCCL::~ProcessGroupNCCL() {
    NCCLCHECK(ncclCommDestroy(comm_)); 
    comm_ = nullptr; 
    CUDACHECK(cudaSetDevice(0)); 
    std::cout << "ProcessGroupNCCL destroyed" << std::endl;
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
    validate_tensor(tensor, device_);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_).stream();

    std::lock_guard<std::mutex> lock(launch_mutex_);

    NCCLCHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        static_cast<size_t>(tensor.numel()),
        to_nccl_dtype(tensor.scalar_type()),
        to_nccl_op(op),
        comm_,
        stream));

    if (!async_op) {
        CUDACHECK(cudaStreamSynchronize(stream));
        return nullptr;
    }

    return WorkNCCL::record(stream, {tensor});
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
