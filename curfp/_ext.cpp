/*
 * curfp/_ext.cpp — pybind11 C++ extension (no torch dependency at build time)
 *
 * Tensors are passed from Python as raw int64 device pointers extracted via
 * tensor.data_ptr().  All validation (dtype, device, contiguity) is done in
 * the Python layer (curfp/__init__.py) before reaching this code.
 *
 * cudaStream_t is likewise passed as an int64 (stream.cuda_stream in Python).
 */

#include <pybind11/pybind11.h>
#include <curfp.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

/* =========================================================================
 * Error helpers
 * =========================================================================*/
static void check_status(curfpStatus_t s)
{
    if (s != CURFP_STATUS_SUCCESS)
        throw std::runtime_error("curfp error: status=" + std::to_string((int)s));
}

/* =========================================================================
 * Handle class
 * =========================================================================*/
class Handle {
    curfpHandle_t h_ = nullptr;
public:
    Handle()  { check_status(curfpCreate(&h_)); }
    ~Handle() { if (h_) curfpDestroy(h_); }

    Handle *enter() { return this; }
    void    exit(py::object, py::object, py::object) {}

    void set_stream_ptr(int64_t ptr) {
        check_status(curfpSetStream(h_, reinterpret_cast<cudaStream_t>(ptr)));
    }

    curfpHandle_t get() const { return h_; }
};

/* =========================================================================
 * ssfrk: symmetric rank-k update in RFP format
 *
 * A_ptr, C_ptr: raw device pointers (int64) from tensor.data_ptr()
 * =========================================================================*/
void ssfrk(Handle  &handle,
           int      transr,
           int      uplo,
           int      trans,
           int      n,
           int      k,
           float    alpha,
           int64_t  A_ptr,
           int      lda,
           float    beta,
           int64_t  C_ptr)
{
    check_status(curfpSsfrk(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        static_cast<curfpOperation_t>(trans),
        n, k,
        &alpha, reinterpret_cast<const float *>(A_ptr), lda,
        &beta,  reinterpret_cast<float *>(C_ptr)));
}

/* =========================================================================
 * spftrf: in-place Cholesky factorization in RFP format
 *
 * A_ptr: raw device pointer (int64) from tensor.data_ptr()
 * Returns info: 0 = success, >0 = not positive definite
 * =========================================================================*/
int spftrf(Handle  &handle,
           int      transr,
           int      uplo,
           int      n,
           int64_t  A_ptr)
{
    int info = 0;
    check_status(curfpSpftrf(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        reinterpret_cast<float *>(A_ptr),
        &info));
    return info;
}

/* =========================================================================
 * Module
 * =========================================================================*/
PYBIND11_MODULE(_curfp_C, m)
{
    m.doc() = "curfp: CUDA RFP format matrix operations";

    py::class_<Handle>(m, "Handle")
        .def(py::init<>())
        .def("__enter__", &Handle::enter, py::return_value_policy::reference)
        .def("__exit__",  &Handle::exit)
        .def("set_stream_ptr", &Handle::set_stream_ptr,
             py::arg("ptr"),
             "Set CUDA stream from raw integer (stream.cuda_stream).");

    m.def("ssfrk",  &ssfrk,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"), py::arg("trans"),
          py::arg("n"),      py::arg("k"),
          py::arg("alpha"),  py::arg("A_ptr"), py::arg("lda"),
          py::arg("beta"),   py::arg("C_ptr"));

    m.def("spftrf", &spftrf,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),      py::arg("A_ptr"));

    m.attr("OP_N")       = static_cast<int>(CURFP_OP_N);
    m.attr("OP_T")       = static_cast<int>(CURFP_OP_T);
    m.attr("FILL_LOWER") = static_cast<int>(CURFP_FILL_MODE_LOWER);
    m.attr("FILL_UPPER") = static_cast<int>(CURFP_FILL_MODE_UPPER);
}
