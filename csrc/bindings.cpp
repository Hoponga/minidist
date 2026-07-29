#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "process_group_nccl.h"
#include "tensor_utils.h"
#include "work_nccl.h"

namespace py = pybind11;

py::bytes get_nccl_unique_id() {
    ncclUniqueId unique_id; 
    ncclGetUniqueId(&unique_id); 

    return py::bytes(reinterpret_cast<char*>(unique_id.internal), sizeof(unique_id));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
        .def(
            py::init<
                const std::string&,
                int,
                int,
                std::vector<int>,
                int>(),
            py::arg("unique_id"),
            py::arg("global_rank"),
            py::arg("group_rank"),
            py::arg("global_ranks"),
            py::arg("device"),
            py::call_guard<py::gil_scoped_release>())
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
