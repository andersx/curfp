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
 * spftrs: triangular solve using RFP Cholesky factor
 *
 * A_ptr: raw device pointer to RFP Cholesky factor (from spftrf)
 * B_ptr: raw device pointer to (n × nrhs) RHS, overwritten with solution
 * =========================================================================*/
void spftrs(Handle  &handle,
            int      transr,
            int      uplo,
            int      n,
            int      nrhs,
            int64_t  A_ptr,
            int64_t  B_ptr,
            int      ldb)
{
    check_status(curfpSpftrs(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n, nrhs,
        reinterpret_cast<const float *>(A_ptr),
        reinterpret_cast<float *>(B_ptr),
        ldb));
}

/* =========================================================================
 * ssfmv: symmetric matrix-vector multiply in RFP format
 *
 * y := alpha * A * x + beta * y
 * arf_ptr, x_ptr, y_ptr: raw device pointers (int64)
 * =========================================================================*/
void ssfmv(Handle        &handle,
           int             transr,
           int             uplo,
           int             n,
           float           alpha,
           int64_t         arf_ptr,
           int64_t         x_ptr,
           int             incx,
           float           beta,
           int64_t         y_ptr,
           int             incy)
{
    check_status(curfpSsfmv(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        &alpha,
        reinterpret_cast<const float *>(arf_ptr),
        reinterpret_cast<const float *>(x_ptr),
        incx,
        &beta,
        reinterpret_cast<float *>(y_ptr),
        incy));
}

/* =========================================================================
 * spftri: compute inverse of SPD matrix from RFP Cholesky factor
 *
 * arf_ptr: raw device pointer to RFP Cholesky factor (from spftrf), overwritten
 * =========================================================================*/
void spftri(Handle  &handle,
            int      transr,
            int      uplo,
            int      n,
            int64_t  arf_ptr)
{
    check_status(curfpSpftri(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        reinterpret_cast<float *>(arf_ptr)));
}

/* =========================================================================
 * strttf: full triangular matrix → RFP format
 *
 * A_ptr: raw device pointer to n×n row-major matrix (int64)
 * arf_ptr: raw device pointer to output RFP array of size n*(n+1)/2 (int64)
 * =========================================================================*/
void strttf(Handle  &handle,
            int      transr,
            int      uplo,
            int      n,
            int64_t  A_ptr,
            int      lda,
            int64_t  arf_ptr)
{
    check_status(curfpSstrttf(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        reinterpret_cast<const float *>(A_ptr),
        lda,
        reinterpret_cast<float *>(arf_ptr)));
}

/* =========================================================================
 * stfttr: RFP format → full triangular matrix
 *
 * arf_ptr: raw device pointer to RFP array of size n*(n+1)/2 (int64)
 * A_ptr: raw device pointer to n×n row-major output matrix (int64)
 * =========================================================================*/
void stfttr(Handle  &handle,
            int      transr,
            int      uplo,
            int      n,
            int64_t  arf_ptr,
            int64_t  A_ptr,
            int      lda)
{
    check_status(curfpSstfttr(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        reinterpret_cast<const float *>(arf_ptr),
        reinterpret_cast<float *>(A_ptr),
        lda));
}

/* =========================================================================
 * slansf: norm of a symmetric matrix in RFP format
 *
 * norm: 0=NORM_MAX, 1=NORM_ONE, 2=NORM_FRO
 * arf_ptr: raw device pointer (int64) from tensor.data_ptr()
 * Returns: norm value as a Python float
 * =========================================================================*/
float slansf(Handle  &handle,
             int      norm,
             int      transr,
             int      uplo,
             int      n,
             int64_t  arf_ptr)
{
    float result = 0.0f;
    check_status(curfpSlansf(
        handle.get(),
        static_cast<curfpNormType_t>(norm),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        reinterpret_cast<const float *>(arf_ptr),
        &result));
    return result;
}

/* =========================================================================
 * spfcon: reciprocal condition number estimate from RFP Cholesky factor
 *
 * arf_ptr: raw device pointer to RFP Cholesky factor (int64)
 * anorm:   1-norm of original matrix (before factorization)
 * Returns: rcond estimate as a Python float
 * =========================================================================*/
float spfcon(Handle  &handle,
             int      transr,
             int      uplo,
             int      n,
             int64_t  arf_ptr,
             float    anorm)
{
    float rcond = 0.0f;
    check_status(curfpSpfcon(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        reinterpret_cast<const float *>(arf_ptr),
        anorm,
        &rcond));
    return rcond;
}

/* =========================================================================
 * ssfr: symmetric rank-1 update in RFP format
 *
 * C := alpha * x * x^T + C
 * arf_ptr, x_ptr: raw device pointers (int64)
 * =========================================================================*/
void ssfr(Handle  &handle,
          int      transr,
          int      uplo,
          int      n,
          float    alpha,
          int64_t  x_ptr,
          int      incx,
          int64_t  arf_ptr)
{
    check_status(curfpSsfr(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        &alpha,
        reinterpret_cast<const float *>(x_ptr),
        incx,
        reinterpret_cast<float *>(arf_ptr)));
}

/* =========================================================================
 * ssfr2: symmetric rank-2 update in RFP format
 *
 * C := alpha * x * y^T + alpha * y * x^T + C
 * arf_ptr, x_ptr, y_ptr: raw device pointers (int64)
 * =========================================================================*/
void ssfr2(Handle  &handle,
           int      transr,
           int      uplo,
           int      n,
           float    alpha,
           int64_t  x_ptr,
           int      incx,
           int64_t  y_ptr,
           int      incy,
           int64_t  arf_ptr)
{
    check_status(curfpSsfr2(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        n,
        &alpha,
        reinterpret_cast<const float *>(x_ptr),
        incx,
        reinterpret_cast<const float *>(y_ptr),
        incy,
        reinterpret_cast<float *>(arf_ptr)));
}

/* =========================================================================
 * ssfr2k: symmetric rank-2k update in RFP format
 *
 * C := alpha*op(A)*op(B)^T + alpha*op(B)*op(A)^T + beta*C
 * A_ptr, B_ptr, C_ptr: raw device pointers (int64)
 * =========================================================================*/
void ssfr2k(Handle  &handle,
            int      transr,
            int      uplo,
            int      trans,
            int      n,
            int      k,
            float    alpha,
            int64_t  A_ptr,
            int      lda,
            int64_t  B_ptr,
            int      ldb,
            float    beta,
            int64_t  C_ptr)
{
    check_status(curfpSsfr2k(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        static_cast<curfpOperation_t>(trans),
        n, k,
        &alpha,
        reinterpret_cast<const float *>(A_ptr), lda,
        reinterpret_cast<const float *>(B_ptr), ldb,
        &beta,
        reinterpret_cast<float *>(C_ptr)));
}

/* =========================================================================
 * ssfmm: symmetric matrix-matrix multiply in RFP format
 *
 * side=LEFT:  C := alpha * A * B + beta * C
 * side=RIGHT: C := alpha * B * A + beta * C
 * arf_ptr, B_ptr, C_ptr: raw device pointers (int64)
 * =========================================================================*/
void ssfmm(Handle  &handle,
           int      transr,
           int      uplo,
           int      side,
           int      m,
           int      n,
           float    alpha,
           int64_t  arf_ptr,
           int64_t  B_ptr,
           int      ldb,
           float    beta,
           int64_t  C_ptr,
           int      ldc)
{
    check_status(curfpSsfmm(
        handle.get(),
        static_cast<curfpOperation_t>(transr),
        static_cast<curfpFillMode_t>(uplo),
        static_cast<curfpSideMode_t>(side),
        m, n,
        &alpha,
        reinterpret_cast<const float *>(arf_ptr),
        reinterpret_cast<const float *>(B_ptr), ldb,
        &beta,
        reinterpret_cast<float *>(C_ptr), ldc));
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

    m.def("ssfmv", &ssfmv,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("alpha"),  py::arg("arf_ptr"),
          py::arg("x_ptr"),  py::arg("incx"),
          py::arg("beta"),   py::arg("y_ptr"), py::arg("incy"));

    m.def("spftri", &spftri,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),      py::arg("arf_ptr"));

    m.def("spftrs", &spftrs,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),      py::arg("nrhs"),
          py::arg("A_ptr"),  py::arg("B_ptr"), py::arg("ldb"));

    m.def("strttf", &strttf,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("A_ptr"),  py::arg("lda"),
          py::arg("arf_ptr"));

    m.def("stfttr", &stfttr,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("arf_ptr"),
          py::arg("A_ptr"),  py::arg("lda"));

    m.def("slansf", &slansf,
          py::arg("handle"),
          py::arg("norm"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("arf_ptr"));

    m.def("spfcon", &spfcon,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("arf_ptr"),
          py::arg("anorm"));

    m.def("ssfr", &ssfr,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("alpha"),  py::arg("x_ptr"), py::arg("incx"),
          py::arg("arf_ptr"));

    m.def("ssfr2", &ssfr2,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"),
          py::arg("n"),
          py::arg("alpha"),
          py::arg("x_ptr"),  py::arg("incx"),
          py::arg("y_ptr"),  py::arg("incy"),
          py::arg("arf_ptr"));

    m.def("ssfr2k", &ssfr2k,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"), py::arg("trans"),
          py::arg("n"),      py::arg("k"),
          py::arg("alpha"),
          py::arg("A_ptr"),  py::arg("lda"),
          py::arg("B_ptr"),  py::arg("ldb"),
          py::arg("beta"),   py::arg("C_ptr"));

    m.def("ssfmm", &ssfmm,
          py::arg("handle"),
          py::arg("transr"), py::arg("uplo"), py::arg("side"),
          py::arg("m"),      py::arg("n"),
          py::arg("alpha"),  py::arg("arf_ptr"),
          py::arg("B_ptr"),  py::arg("ldb"),
          py::arg("beta"),   py::arg("C_ptr"), py::arg("ldc"));

    m.attr("OP_N")        = static_cast<int>(CURFP_OP_N);
    m.attr("OP_T")        = static_cast<int>(CURFP_OP_T);
    m.attr("FILL_LOWER")  = static_cast<int>(CURFP_FILL_MODE_LOWER);
    m.attr("FILL_UPPER")  = static_cast<int>(CURFP_FILL_MODE_UPPER);
    m.attr("NORM_MAX")    = static_cast<int>(CURFP_NORM_MAX);
    m.attr("NORM_ONE")    = static_cast<int>(CURFP_NORM_ONE);
    m.attr("NORM_FRO")    = static_cast<int>(CURFP_NORM_FRO);
    m.attr("SIDE_LEFT")   = static_cast<int>(CURFP_SIDE_LEFT);
    m.attr("SIDE_RIGHT")  = static_cast<int>(CURFP_SIDE_RIGHT);
}
