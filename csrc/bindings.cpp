#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "process_group_nccl.h"
#include "tensor_utils.h"
#include "work_nccl.h"

namespace py = pybind11;

std::string get_nccl_unique_id() {
}

PYBIND11_MODULE(_C, m) {
    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("SUM", ReduceOp::SUM)
        .value("PRODUCT", ReduceOp::PRODUCT)
        .value("MIN", ReduceOp::MIN)
        .value("MAX", ReduceOp::MAX)
        .value("AVG", ReduceOp::AVG);

    m.def("get_nccl_unique_id", &get_nccl_unique_id);

    py::class_<WorkNCCL, std::shared_ptr<WorkNCCL>>(m, "WorkNCCL")
        .def("is_completed", &WorkNCCL::is_completed)
        .def("wait", &WorkNCCL::wait);

    py::class_<ProcessGroupNCCL, std::shared_ptr<ProcessGroupNCCL>>(
        m, "ProcessGroupNCCL")
        .def(py::init<
             const std::string&,
             int,
             int,
             std::vector<int>,
             int>())
        .def("rank", &ProcessGroupNCCL::rank)
        .def("size", &ProcessGroupNCCL::size)
        .def("all_reduce", &ProcessGroupNCCL::all_reduce)
        .def("broadcast", &ProcessGroupNCCL::broadcast)
        .def("reduce", &ProcessGroupNCCL::reduce)
        .def(
            "all_gather_into_tensor",
            &ProcessGroupNCCL::all_gather_into_tensor)
        .def(
            "reduce_scatter_tensor",
            &ProcessGroupNCCL::reduce_scatter_tensor)
        .def("barrier", &ProcessGroupNCCL::barrier)
        .def("destroy", &ProcessGroupNCCL::destroy);
}
