#pragma once

#include <torch/extension.h>
#include <nccl.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tensor_utils.h"
#include "work_nccl.h"

class ProcessGroupNCCL {
public:
    ProcessGroupNCCL(
        const std::string& unique_id_bytes,
        int global_rank,
        int group_rank,
        std::vector<int> global_ranks,
        int device);

    ~ProcessGroupNCCL();

    int rank() const;
    int size() const;

    std::shared_ptr<WorkNCCL> all_reduce(
        torch::Tensor tensor,
        ReduceOp op,
        bool async_op);

    std::shared_ptr<WorkNCCL> broadcast(
        torch::Tensor tensor,
        int global_src,
        bool async_op);

    std::shared_ptr<WorkNCCL> reduce(
        torch::Tensor tensor,
        int global_dst,
        ReduceOp op,
        bool async_op);

    std::shared_ptr<WorkNCCL> all_gather_into_tensor(
        torch::Tensor output,
        torch::Tensor input,
        bool async_op);

    std::shared_ptr<WorkNCCL> reduce_scatter_tensor(
        torch::Tensor output,
        torch::Tensor input,
        ReduceOp op,
        bool async_op);

    void barrier();
    void destroy();

private:
    int global_rank_;
    int group_rank_;
    int device_;

    std::vector<int> global_ranks_;
    ncclComm_t comm_ = nullptr;

    std::mutex launch_mutex_;

    int to_group_rank(int global_rank) const;
};
