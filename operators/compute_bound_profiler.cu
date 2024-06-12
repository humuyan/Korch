#include <pybind11/pybind11.h>

#include "conv.h"
#include "gemm.h"

namespace py = pybind11;

PYBIND11_MODULE(compute_bound_profiler, m) {
    m.def("profile_conv",
          &profile_conv,
          py::arg("batch_size"),
          py::arg("in_channels"),
          py::arg("in_height"),
          py::arg("in_width"),
          py::arg("out_channels"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"),
          py::arg("algo"),
          py::arg("mode"));  // 0 - no bias/relu, 1 - with bias no relu,
                             // 2 - with bias/relu
    m.def("profile_gemm",
          &profile_gemm,
          py::arg("b"),
          py::arg("m"),
          py::arg("n"),
          py::arg("k"),
          py::arg("transa"),
          py::arg("transb"),
          py::arg("tf32"));
}