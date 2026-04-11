"""
Benchmark: curfp double-precision (D-prefix, RFP) vs PyTorch/cuBLAS/cuSOLVER (dense).

Identical structure to benchmark.py but for float64.  Sizes capped at 4096
(double uses 2× memory).  No cuBLAS Dspmv / Dstrttp / Dstpttr (those packed
routines have no double-precision equivalents in cuBLAS), so those comparison
lines are omitted from the ssfmv / strttf / stfttr plots.

Colour convention (same as benchmark.py):
  Blue  (#2166ac / #6baed6) — curfp RFP
  Green (#238b45 / #74c476 / #005a32) — cuBLAS / cuSOLVER
  Red   (#cb181d / #fb6a4a) — PyTorch / torch
"""

import csv
import ctypes

import torch
import curfp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load libraries
# ---------------------------------------------------------------------------
_libcublas = ctypes.CDLL("libcublas.so.12", use_errno=True)
_libcusolver = ctypes.CDLL("libcusolver.so.11", use_errno=True)

# ---------------------------------------------------------------------------
# cuBLAS handle
# ---------------------------------------------------------------------------
_libcublas.cublasCreate_v2.restype = ctypes.c_int
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcublas.cublasDestroy_v2.restype = ctypes.c_int
_libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

_cublas_handle = ctypes.c_void_p()
assert _libcublas.cublasCreate_v2(ctypes.byref(_cublas_handle)) == 0

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_SIDE_LEFT = 0
CUBLAS_SIDE_RIGHT = 1

# ---------------------------------------------------------------------------
# cublasDgemm  (dense A*A^T: OP_N × OP_T)
# ---------------------------------------------------------------------------
_libcublas.cublasDgemm_v2.restype = ctypes.c_int
_libcublas.cublasDgemm_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dgemm(n, k, alpha, A, lda, beta, C, ldc):
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    ret = _libcublas.cublasDgemm_v2(
        _cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        n,
        n,
        k,
        ctypes.byref(_alpha),
        ctypes.c_void_p(A.data_ptr()),
        lda,
        ctypes.c_void_p(A.data_ptr()),
        lda,
        ctypes.byref(_beta),
        ctypes.c_void_p(C.data_ptr()),
        ldc,
    )
    assert ret == 0, f"cublasDgemm_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasDsyrk  (OP_T: A is k×n col-major lda=k → A^T*A = n×n)
# ---------------------------------------------------------------------------
_libcublas.cublasDsyrk_v2.restype = ctypes.c_int
_libcublas.cublasDsyrk_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dsyrk(n, k, alpha, A, lda, beta, C, ldc):
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    ret = _libcublas.cublasDsyrk_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        n,
        k,
        ctypes.byref(_alpha),
        ctypes.c_void_p(A.data_ptr()),
        lda,
        ctypes.byref(_beta),
        ctypes.c_void_p(C.data_ptr()),
        ldc,
    )
    assert ret == 0, f"cublasDsyrk_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasDsymv
# ---------------------------------------------------------------------------
_libcublas.cublasDsymv_v2.restype = ctypes.c_int
_libcublas.cublasDsymv_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dsymv(n, A, x, y):
    _alpha = ctypes.c_double(1.0)
    _beta = ctypes.c_double(0.0)
    ret = _libcublas.cublasDsymv_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        ctypes.byref(_alpha),
        ctypes.c_void_p(A.data_ptr()),
        n,
        ctypes.c_void_p(x.data_ptr()),
        1,
        ctypes.byref(_beta),
        ctypes.c_void_p(y.data_ptr()),
        1,
    )
    assert ret == 0, f"cublasDsymv_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasDsyr
# ---------------------------------------------------------------------------
_libcublas.cublasDsyr_v2.restype = ctypes.c_int
_libcublas.cublasDsyr_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dsyr(n, x, A):
    _alpha = ctypes.c_double(1.0)
    ret = _libcublas.cublasDsyr_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        ctypes.byref(_alpha),
        ctypes.c_void_p(x.data_ptr()),
        1,
        ctypes.c_void_p(A.data_ptr()),
        n,
    )
    assert ret == 0, f"cublasDsyr_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasDsyr2
# ---------------------------------------------------------------------------
_libcublas.cublasDsyr2_v2.restype = ctypes.c_int
_libcublas.cublasDsyr2_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dsyr2(n, x, y, A):
    _alpha = ctypes.c_double(1.0)
    ret = _libcublas.cublasDsyr2_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        ctypes.byref(_alpha),
        ctypes.c_void_p(x.data_ptr()),
        1,
        ctypes.c_void_p(y.data_ptr()),
        1,
        ctypes.c_void_p(A.data_ptr()),
        n,
    )
    assert ret == 0, f"cublasDsyr2_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasDsyr2k  (NOTRANS: A and B are n×k col-major lda=n)
# ---------------------------------------------------------------------------
_libcublas.cublasDsyr2k_v2.restype = ctypes.c_int
_libcublas.cublasDsyr2k_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dsyr2k(n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    ret = _libcublas.cublasDsyr2k_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        n,
        k,
        ctypes.byref(_alpha),
        ctypes.c_void_p(A.data_ptr()),
        lda,
        ctypes.c_void_p(B.data_ptr()),
        ldb,
        ctypes.byref(_beta),
        ctypes.c_void_p(C.data_ptr()),
        ldc,
    )
    assert ret == 0, f"cublasDsyr2k_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasDsymm  (side=LEFT, lower tri, col-major)
# ---------------------------------------------------------------------------
_libcublas.cublasDsymm_v2.restype = ctypes.c_int
_libcublas.cublasDsymm_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_dsymm(m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    ret = _libcublas.cublasDsymm_v2(
        _cublas_handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        m,
        n,
        ctypes.byref(_alpha),
        ctypes.c_void_p(A.data_ptr()),
        lda,
        ctypes.c_void_p(B.data_ptr()),
        ldb,
        ctypes.byref(_beta),
        ctypes.c_void_p(C.data_ptr()),
        ldc,
    )
    assert ret == 0, f"cublasDsymm_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cuSOLVER handle + dpotrf / dpotri
# ---------------------------------------------------------------------------
_libcusolver.cusolverDnCreate.restype = ctypes.c_int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcusolver.cusolverDnDestroy.restype = ctypes.c_int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverDnDpotrf_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnDpotrf_bufferSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_libcusolver.cusolverDnDpotrf.restype = ctypes.c_int
_libcusolver.cusolverDnDpotrf.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]

_libcusolver.cusolverDnDpotri_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnDpotri_bufferSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_libcusolver.cusolverDnDpotri.restype = ctypes.c_int
_libcusolver.cusolverDnDpotri.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]

CUSOLVER_FILL_MODE_LOWER = 0
_cusolver_handle = ctypes.c_void_p()
assert _libcusolver.cusolverDnCreate(ctypes.byref(_cusolver_handle)) == 0

_cusolver_workspace = {}
_cusolver_potri_workspace = {}


def cusolver_dpotrf(n, S):
    if n not in _cusolver_workspace:
        lwork = ctypes.c_int(0)
        ret = _libcusolver.cusolverDnDpotrf_bufferSize(
            _cusolver_handle,
            CUSOLVER_FILL_MODE_LOWER,
            n,
            ctypes.c_void_p(S.data_ptr()),
            n,
            ctypes.byref(lwork),
        )
        if ret != 0:
            raise RuntimeError(f"cusolverDn bufferSize returned {ret}")
        workspace = torch.empty(lwork.value, dtype=torch.float64, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")
        _cusolver_workspace[n] = (workspace, devInfo, lwork.value)
    workspace, devInfo, lwork = _cusolver_workspace[n]
    ret = _libcusolver.cusolverDnDpotrf(
        _cusolver_handle,
        CUSOLVER_FILL_MODE_LOWER,
        n,
        ctypes.c_void_p(S.data_ptr()),
        n,
        ctypes.c_void_p(workspace.data_ptr()),
        lwork,
        ctypes.c_void_p(devInfo.data_ptr()),
    )
    if ret != 0:
        raise RuntimeError(f"cusolverDnDpotrf returned {ret}")
    return devInfo.item()


def cusolver_dpotri(n, S):
    if n not in _cusolver_potri_workspace:
        lwork = ctypes.c_int(0)
        ret = _libcusolver.cusolverDnDpotri_bufferSize(
            _cusolver_handle,
            CUSOLVER_FILL_MODE_LOWER,
            n,
            ctypes.c_void_p(S.data_ptr()),
            n,
            ctypes.byref(lwork),
        )
        if ret != 0:
            raise RuntimeError(f"cusolverDn bufferSize returned {ret}")
        workspace = torch.empty(lwork.value, dtype=torch.float64, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")
        _cusolver_potri_workspace[n] = (workspace, devInfo, lwork.value)
    workspace, devInfo, lwork = _cusolver_potri_workspace[n]
    ret = _libcusolver.cusolverDnDpotri(
        _cusolver_handle,
        CUSOLVER_FILL_MODE_LOWER,
        n,
        ctypes.c_void_p(S.data_ptr()),
        n,
        ctypes.c_void_p(workspace.data_ptr()),
        lwork,
        ctypes.c_void_p(devInfo.data_ptr()),
    )
    if ret != 0:
        raise RuntimeError(f"cusolverDnDpotri returned {ret}")
    return devInfo.item()


# ---------------------------------------------------------------------------
# Timing helper  (identical to benchmark.py)
# ---------------------------------------------------------------------------
def cuda_time(setup_fn, fn, warmup=1, repeat=3):
    """Return median elapsed milliseconds over `repeat` timed runs."""
    for _ in range(warmup):
        setup_fn()
        fn()
    torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(repeat):
        setup_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], result


# ---------------------------------------------------------------------------
# Pipeline benchmarks: dsfrk / dpftrf / dpftrs
# ---------------------------------------------------------------------------


def bench_curfp(n, k, nrhs):
    mem_gb = n * (n + 1) // 2 * 8 / 1024**3
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float64, device="cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None, mem_gb

    t_sfrk, _ = cuda_time(lambda: None, lambda: curfp.dsfrk(A, C))

    curfp.add_to_diagonal(C, 1.0)
    t_chol, _ = cuda_time(
        lambda: (curfp.dsfrk(A, C), curfp.add_to_diagonal(C, 1.0)),
        lambda: curfp.dpftrf(C, check=False),
    )
    t_solve, _ = cuda_time(
        lambda: (
            curfp.dsfrk(A, C),
            curfp.add_to_diagonal(C, 1.0),
            curfp.dpftrf(C, check=False),
        ),
        lambda: curfp.dpftrs(C, B),
    )

    mem_gb = C.numel() * C.element_size() / 1024**3
    del A, C, B
    return t_sfrk, t_chol, t_solve, mem_gb


def bench_torch(n, k, nrhs):
    mat_gb = n * n * 8 / 1024**3
    t_torch_dgemm = t_cublas_dgemm = t_dsyrk = None
    t_torch_chol = t_cusolver_chol = t_solve = None

    try:
        A = (torch.randn(k, n, dtype=torch.float64, device="cuda") / k**0.5).t()
        S = torch.empty(n, n, dtype=torch.float64, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float64, device="cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return (
            t_torch_dgemm,
            t_cublas_dgemm,
            t_dsyrk,
            t_torch_chol,
            t_cusolver_chol,
            t_solve,
            mat_gb,
        )

    t_torch_dgemm, _ = cuda_time(lambda: None, lambda: torch.mm(A, A.t(), out=S))
    t_cublas_dgemm, _ = cuda_time(
        lambda: None, lambda: cublas_dgemm(n, k, 1.0, A, n, 0.0, S, n)
    )
    # A is (n,k) row-major = k×n col-major; OP_T with lda=k → A^T*A (n×n)
    t_dsyrk, _ = cuda_time(
        lambda: None, lambda: cublas_dsyrk(n, k, 1.0, A, k, 0.0, S, n)
    )

    def dsyrk_diag():
        cublas_dsyrk(n, k, 1.0, A, k, 0.0, S, n)
        S.diagonal().add_(1.0)

    def dgemm_diag():
        torch.mm(A, A.t(), out=S)
        S.diagonal().add_(1.0)

    try:
        t_cusolver_chol, info = cuda_time(dsyrk_diag, lambda: cusolver_dpotrf(n, S))
        assert info == 0
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    try:
        t_torch_chol, _ = cuda_time(dgemm_diag, lambda: torch.linalg.cholesky(S, out=S))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    if t_torch_chol is not None:

        def restore_chol():
            torch.mm(A, A.t(), out=S)
            S.diagonal().add_(1.0)
            torch.linalg.cholesky(S, out=S)

        try:
            t_solve, _ = cuda_time(restore_chol, lambda: torch.cholesky_solve(B, S))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    mem_gb = S.numel() * S.element_size() / 1024**3
    del A, S, B
    return (
        t_torch_dgemm,
        t_cublas_dgemm,
        t_dsyrk,
        t_torch_chol,
        t_cusolver_chol,
        t_solve,
        mat_gb,
    )


# ---------------------------------------------------------------------------
# dpftri
# ---------------------------------------------------------------------------


def bench_dpftri(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")

        def setup():
            curfp.dsfrk(A, C)
            curfp.add_to_diagonal(C, 1.0)
            curfp.dpftrf(C, check=False)

        t_rfp, _ = cuda_time(setup, lambda: curfp.dpftri(C))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_potri = t_inv = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.zeros(n, n, dtype=torch.float64, device="cuda")

        def setup_chol():
            torch.mm(A2, A2.t(), out=S)
            S.diagonal().add_(1.0)
            cusolver_dpotrf(n, S)

        setup_chol()
        t_potri, _ = cuda_time(setup_chol, lambda: cusolver_dpotri(n, S))

        def setup_plain():
            torch.mm(A2, A2.t(), out=S)
            S.diagonal().add_(1.0)

        t_inv, _ = cuda_time(setup_plain, lambda: torch.linalg.inv(S))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_potri, t_inv


# ---------------------------------------------------------------------------
# dlansf
# ---------------------------------------------------------------------------


def bench_dlansf(n, k):
    torch.cuda.empty_cache()
    t_1 = t_fro = None
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_1, _ = cuda_time(lambda: None, lambda: curfp.dlansf(C, "1"))
        t_fro, _ = cuda_time(lambda: None, lambda: curfp.dlansf(C, "F"))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_d1 = t_dfro = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        t_d1, _ = cuda_time(lambda: None, lambda: torch.linalg.matrix_norm(S, 1))
        t_dfro, _ = cuda_time(lambda: None, lambda: torch.linalg.matrix_norm(S, "fro"))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_1, t_fro, t_d1, t_dfro


# ---------------------------------------------------------------------------
# dsfmv  (no Dspmv in cuBLAS, so SP packed comparison is omitted)
# ---------------------------------------------------------------------------


def bench_dsfmv(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        x = torch.randn(n, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.dsfmv(C, x))
        del A, C, x
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_symv = t_mv = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float64, device="cuda")
        y_symv = torch.empty(n, dtype=torch.float64, device="cuda")

        t_symv, _ = cuda_time(lambda: None, lambda: cublas_dsymv(n, S, x2, y_symv))
        t_mv, _ = cuda_time(lambda: None, lambda: torch.mv(S, x2))

        del A2, S, x2, y_symv
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_symv, t_mv


# ---------------------------------------------------------------------------
# dstrttf  (no Dstrttp in cuBLAS, so TP packed comparison is omitted)
# ---------------------------------------------------------------------------


def bench_dstrttf(n, k):
    torch.cuda.empty_cache()
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A, A.t())
        S.diagonal().add_(1.0)
        tri = torch.triu(S)
        del S, A
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None, None

    t_rfp = t_dense = None
    try:
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.dstrttf(tri, uplo="U"))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    try:
        t_dense, _ = cuda_time(lambda: None, lambda: torch.triu(tri))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass

    del tri
    torch.cuda.empty_cache()
    return t_rfp, t_dense


# ---------------------------------------------------------------------------
# dstfttr  (no Dstpttr in cuBLAS, so TP packed comparison is omitted)
# ---------------------------------------------------------------------------


def bench_dstfttr(n, k):
    torch.cuda.empty_cache()
    t_rfp = t_dense = None

    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.dstfttr(C, uplo="U"))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        t_dense, _ = cuda_time(lambda: None, lambda: torch.tril(S))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dense


# ---------------------------------------------------------------------------
# dsfr
# ---------------------------------------------------------------------------


def bench_dsfr(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        x = torch.randn(n, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.dsfr(C, x))
        del A, C, x
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dsyr = t_addr = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float64, device="cuda")
        t_dsyr, _ = cuda_time(lambda: None, lambda: cublas_dsyr(n, x2, S))
        t_addr, _ = cuda_time(lambda: None, lambda: torch.addr(S, x2, x2, out=S))
        del A2, S, x2
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dsyr, t_addr


# ---------------------------------------------------------------------------
# dsfr2
# ---------------------------------------------------------------------------


def bench_dsfr2(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        x = torch.randn(n, dtype=torch.float64, device="cuda")
        y = torch.randn(n, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.dsfr2(C, x, y))
        del A, C, x, y
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dsyr2 = t_addr2 = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float64, device="cuda")
        y2 = torch.randn(n, dtype=torch.float64, device="cuda")
        t_dsyr2, _ = cuda_time(lambda: None, lambda: cublas_dsyr2(n, x2, y2, S))
        t_addr2, _ = cuda_time(
            lambda: None,
            lambda: (torch.addr(S, x2, y2, out=S), torch.addr(S, y2, x2, out=S)),
        )
        del A2, S, x2, y2
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dsyr2, t_addr2


# ---------------------------------------------------------------------------
# dsfr2k
# ---------------------------------------------------------------------------


def bench_dsfr2k(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A_rfp = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        B_rfp = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C_rfp = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A_rfp, C_rfp)
        curfp.add_to_diagonal(C_rfp, 1.0)
        t_rfp, _ = cuda_time(
            lambda: None,
            lambda: curfp.dsfr2k(A_rfp, B_rfp, C_rfp, beta=1.0),
        )
        del A_rfp, B_rfp, C_rfp
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dsyr2k = t_gemm2 = None
    try:
        A_dns = torch.randn(k, n, dtype=torch.float64, device="cuda").t().contiguous()
        B_dns = torch.randn(k, n, dtype=torch.float64, device="cuda").t().contiguous()
        S = torch.zeros(n, n, dtype=torch.float64, device="cuda")
        torch.mm(A_dns, A_dns.t(), out=S)
        S.diagonal().add_(1.0)
        t_dsyr2k, _ = cuda_time(
            lambda: None,
            lambda: cublas_dsyr2k(n, k, 1.0, A_dns, n, B_dns, n, 1.0, S, n),
        )

        def two_dgemm():
            cublas_dgemm(n, k, 1.0, A_dns, n, 0.0, S, n)
            cublas_dgemm(n, k, 1.0, B_dns, n, 1.0, S, n)

        t_gemm2, _ = cuda_time(lambda: None, two_dgemm)
        del A_dns, B_dns, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dsyr2k, t_gemm2


# ---------------------------------------------------------------------------
# dsfmm
# ---------------------------------------------------------------------------


def bench_dsfmm(n, k, nrhs):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A_rfp = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C_rfp = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        # B and out as (nrhs, n) row-major; dsfmm_raw convention: m=n, n_p=nrhs, ldb=n
        B_rfp = torch.randn(nrhs, n, dtype=torch.float64, device="cuda")
        out_rfp = torch.empty(nrhs, n, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A_rfp, C_rfp)
        curfp.add_to_diagonal(C_rfp, 1.0)
        h = curfp.Handle()
        t_rfp, _ = cuda_time(
            lambda: None,
            lambda: curfp.dsfmm_raw(
                h,
                curfp.OP_T,
                curfp.FILL_UPPER,
                curfp.SIDE_LEFT,
                n,
                nrhs,
                1.0,
                C_rfp,
                B_rfp,
                n,
                0.0,
                out_rfp,
                n,
            ),
        )
        del A_rfp, C_rfp, B_rfp, out_rfp
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dsymm = t_mm = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        B_dns = torch.randn(n, nrhs, dtype=torch.float64, device="cuda")
        C_dns = torch.empty(n, nrhs, dtype=torch.float64, device="cuda")
        t_dsymm, _ = cuda_time(
            lambda: None,
            lambda: cublas_dsymm(n, nrhs, 1.0, S, n, B_dns, n, 0.0, C_dns, n),
        )
        t_mm, _ = cuda_time(lambda: None, lambda: torch.mm(S, B_dns))
        del A2, S, B_dns, C_dns
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dsymm, t_mm


# ---------------------------------------------------------------------------
# dpfcon
# ---------------------------------------------------------------------------


def bench_dpfcon(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float64, device="cuda")
        curfp.dsfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        anorm = curfp.dlansf(C, "1")
        curfp.dpftrf(C, check=False)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.dpfcon(C, anorm))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dense = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float64, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        t_dense, _ = cuda_time(lambda: None, lambda: torch.linalg.cond(S, 1))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dense


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms1(v, w=9):
    return f"{v:>{w}.1f}ms" if v is not None else f"{'OOM':>{w + 2}}"


def _ms2(v):
    return f"{v:>8.2f}ms" if v is not None else f"{'OOM':>10}"


def _nn(results, idx):
    return [(r[0], r[idx]) for r in results if r[idx] is not None]


# ---------------------------------------------------------------------------
# Colour / style palette
# ---------------------------------------------------------------------------
C_RFP_BLUE = "#2166ac"
C_RFP_BLUE2 = "#6baed6"
C_CB_GREEN = "#238b45"
C_CB_GREEN2 = "#74c476"
C_CS_GREEN = "#005a32"
C_PT_RED = "#cb181d"
C_PT_RED2 = "#fb6a4a"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A100 80 GB — RFP float64 ceiling ~n=130k; dense OOMs expected above ~n=32k
    # sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 122880]
    # H200 141 GB — RFP float64 ceiling ~n=194k; dense OOMs expected above ~n=56k
    # sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 163840, 188416]
    sizes = [1024, 2048, 4096]
    k = 512  # half of benchmark.py's 1024 (2× element size)
    nrhs = 64
    nrhs_mm = 1024

    print(
        f"Benchmark: curfp D-prefix (RFP) vs dense — all ops, float64, rank-{k} update, {nrhs} rhs"
    )

    # ---- Table 1: pipeline ---------------------------------------------------
    print()
    print(
        f"{'n':>6}  {'curfp dsfrk':>12} {'curfp dpftrf':>13} {'curfp dpftrs':>13} {'curfp mem':>10}"
        f"  {'dsyrk':>8} {'cb dgemm':>9} {'pt dgemm':>9} {'cusol chol':>11} {'torch chol':>11} {'torch solve':>12} {'dense mem':>10}"
    )
    print("-" * 170)

    pipeline_results = []

    for n in sizes:
        t_sfrk, t_chol_rfp, t_solve_rfp, mem_rfp = bench_curfp(n, k, nrhs)
        torch.cuda.empty_cache()
        (
            t_pt_dgemm,
            t_cb_dgemm,
            t_dsyrk,
            t_torch_chol,
            t_cusolver_chol,
            t_solve_dns,
            mem_dns,
        ) = bench_torch(n, k, nrhs)
        torch.cuda.empty_cache()

        pipeline_results.append(
            (
                n,
                t_sfrk,
                t_chol_rfp,
                t_solve_rfp,
                mem_rfp,
                t_dsyrk,
                t_cb_dgemm,
                t_pt_dgemm,
                t_cusolver_chol,
                t_torch_chol,
                t_solve_dns,
                mem_dns,
            )
        )

        print(
            f"{n:>6}"
            f"  {_ms1(t_sfrk, 10)}"
            f"  {_ms1(t_chol_rfp, 11)}"
            f"  {_ms1(t_solve_rfp, 11)}"
            f"  {mem_rfp:>8.3f} GB"
            f"  {_ms1(t_dsyrk)}"
            f"  {_ms1(t_cb_dgemm)}"
            f"  {_ms1(t_pt_dgemm)}"
            f"  {_ms1(t_cusolver_chol)}"
            f"  {_ms1(t_torch_chol)}"
            f"  {_ms1(t_solve_dns)}"
            f"  {mem_dns:>8.3f} GB"
        )

    # ---- Table 2: remaining ops ----------------------------------------------
    print()
    print(
        f"{'n':>6}  {'dpftri rfp':>11} {'potri dns':>10} {'inv dns':>9}"
        f"  {'dlansf 1':>9} {'dlansf F':>9} {'norm1 dns':>10} {'normF dns':>10}"
        f"  {'dsfmv rfp':>10} {'symv dns':>9} {'mv dense':>9}"
        f"  {'dstrttf':>8} {'triu dns':>9}"
        f"  {'dstfttr':>8} {'tril dns':>9}"
        f"  {'dpfcon':>8} {'cond dns':>9}"
    )
    print("-" * 210)

    extra_results = []

    for n in sizes:
        t_dpftri_r, t_potri_d, t_inv_d = bench_dpftri(n, k)
        torch.cuda.empty_cache()
        t_s1, t_sf, t_d1, t_df = bench_dlansf(n, k)
        torch.cuda.empty_cache()
        t_dsfmv_r, t_symv_d, t_mv_d = bench_dsfmv(n, k)
        torch.cuda.empty_cache()
        t_dstrttf_r, t_dstrttf_d = bench_dstrttf(n, k)
        torch.cuda.empty_cache()
        t_dstfttr_r, t_dstfttr_d = bench_dstfttr(n, k)
        torch.cuda.empty_cache()
        t_dpfcon_r, t_dpfcon_d = bench_dpfcon(n, k)
        torch.cuda.empty_cache()

        # Index map:
        #  0:n
        #  1:dpftri_r   2:potri_d    3:inv_d
        #  4:s1         5:sf         6:d1         7:df
        #  8:dsfmv_r    9:symv_d    10:mv_d
        # 11:dstrttf_r 12:dstrttf_d
        # 13:dstfttr_r 14:dstfttr_d
        # 15:dpfcon_r  16:dpfcon_d
        extra_results.append(
            (
                n,
                t_dpftri_r,
                t_potri_d,
                t_inv_d,
                t_s1,
                t_sf,
                t_d1,
                t_df,
                t_dsfmv_r,
                t_symv_d,
                t_mv_d,
                t_dstrttf_r,
                t_dstrttf_d,
                t_dstfttr_r,
                t_dstfttr_d,
                t_dpfcon_r,
                t_dpfcon_d,
            )
        )

        print(
            f"{n:>6}"
            f"  {_ms2(t_dpftri_r)} {_ms2(t_potri_d)} {_ms2(t_inv_d)}"
            f"  {_ms2(t_s1)} {_ms2(t_sf)} {_ms2(t_d1)} {_ms2(t_df)}"
            f"  {_ms2(t_dsfmv_r)} {_ms2(t_symv_d)} {_ms2(t_mv_d)}"
            f"  {_ms2(t_dstrttf_r)} {_ms2(t_dstrttf_d)}"
            f"  {_ms2(t_dstfttr_r)} {_ms2(t_dstfttr_d)}"
            f"  {_ms2(t_dpfcon_r)} {_ms2(t_dpfcon_d)}"
        )

    # ---- Table 3: new RFP ops ------------------------------------------------
    print()
    print(
        f"{'n':>6}"
        f"  {'dsfr rfp':>9} {'dsyr dns':>9} {'addr dns':>9}"
        f"  {'dsfr2 rfp':>10} {'dsyr2 dns':>10} {'addr2 dns':>10}"
        f"  {'dsfr2k rfp':>11} {'dsyr2k dns':>11} {'gemm2 dns':>10}"
        f"  {'dsfmm rfp':>10} {'dsymm dns':>10} {'mm dns':>8}  (dsfmm nrhs={nrhs_mm})"
    )
    print("-" * 175)

    new_results = []

    for n in sizes:
        t_dsfr_r, t_dsyr_d, t_addr_d = bench_dsfr(n, k)
        torch.cuda.empty_cache()
        t_dsfr2_r, t_dsyr2_d, t_addr2_d = bench_dsfr2(n, k)
        torch.cuda.empty_cache()
        t_dsfr2k_r, t_dsyr2k_d, t_gemm2_d = bench_dsfr2k(n, k)
        torch.cuda.empty_cache()
        t_dsfmm_r, t_dsymm_d, t_mm_d = bench_dsfmm(n, k, nrhs_mm)
        torch.cuda.empty_cache()

        # Index map:
        #  0:n
        #  1:dsfr_r    2:dsyr_d    3:addr_d
        #  4:dsfr2_r   5:dsyr2_d   6:addr2_d
        #  7:dsfr2k_r  8:dsyr2k_d  9:gemm2_d
        # 10:dsfmm_r  11:dsymm_d  12:mm_d
        new_results.append(
            (
                n,
                t_dsfr_r,
                t_dsyr_d,
                t_addr_d,
                t_dsfr2_r,
                t_dsyr2_d,
                t_addr2_d,
                t_dsfr2k_r,
                t_dsyr2k_d,
                t_gemm2_d,
                t_dsfmm_r,
                t_dsymm_d,
                t_mm_d,
            )
        )

        print(
            f"{n:>6}"
            f"  {_ms2(t_dsfr_r)} {_ms2(t_dsyr_d)} {_ms2(t_addr_d)}"
            f"  {_ms2(t_dsfr2_r)} {_ms2(t_dsyr2_d)} {_ms2(t_addr2_d)}"
            f"  {_ms2(t_dsfr2k_r)} {_ms2(t_dsyr2k_d)} {_ms2(t_gemm2_d)}"
            f"  {_ms2(t_dsfmm_r)} {_ms2(t_dsymm_d)} {_ms2(t_mm_d)}"
        )

    # -------------------------------------------------------------------------
    # Plot: 3 rows × 5 columns  (identical layout to benchmark.py)
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(3, 5, figsize=(30, 18))
    fig.suptitle(
        f"curfp D-prefix (RFP) vs dense — all ops, float64, rank-{k} update, {nrhs} rhs",
        fontsize=14,
    )

    ns_all = [r[0] for r in pipeline_results]

    def style_ax(ax, title, ylabel="Time (ms)"):
        ax.set_xlabel("Matrix size n")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns_all)
        ax.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    # ---- [0,0] dsfrk ----------------------------------------------------------
    ax = axes[0, 0]
    pairs = _nn(pipeline_results, 1)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dsfrk (RFP)")
    pairs = _nn(pipeline_results, 5)
    if pairs:
        ax.plot(
            *zip(*pairs), "s--", color=C_CB_GREEN, label="cublas dsyrk (dense lower)"
        )
    pairs = _nn(pipeline_results, 6)
    if pairs:
        ax.plot(
            *zip(*pairs), "^--", color=C_CB_GREEN2, label="cublas dgemm (dense full)"
        )
    pairs = _nn(pipeline_results, 7)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch mm (dense full)")
    style_ax(ax, "dsfrk — rank-k update")

    # ---- [0,1] dpftrf ----------------------------------------------------------
    ax = axes[0, 1]
    pairs = _nn(pipeline_results, 2)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dpftrf (RFP)")
    pairs = _nn(pipeline_results, 8)
    if pairs:
        ax.plot(*zip(*pairs), "D-.", color=C_CS_GREEN, label="cusolver dpotrf (dense)")
    pairs = _nn(pipeline_results, 9)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch cholesky (dense)")
    style_ax(ax, "dpftrf — Cholesky factorization")

    # ---- [0,2] dpftrs ----------------------------------------------------------
    ax = axes[0, 2]
    pairs = _nn(pipeline_results, 3)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "o-",
            color=C_RFP_BLUE,
            label=f"curfp dpftrs (RFP, {nrhs} rhs)",
        )
    pairs = _nn(pipeline_results, 10)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^:",
            color=C_PT_RED,
            label=f"torch.cholesky_solve ({nrhs} rhs)",
        )
    style_ax(ax, "dpftrs — triangular solve")

    # ---- [0,3] dpftri ----------------------------------------------------------
    ax = axes[0, 3]
    pairs = _nn(extra_results, 1)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dpftri (RFP)")
    pairs = _nn(extra_results, 2)
    if pairs:
        ax.plot(*zip(*pairs), "D-.", color=C_CS_GREEN, label="cusolver dpotri (dense)")
    pairs = _nn(extra_results, 3)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.linalg.inv (dense LU)")
    style_ax(ax, "dpftri — matrix inversion")

    # ---- [0,4] memory ----------------------------------------------------------
    ax = axes[0, 4]
    ax.plot(
        ns_all,
        [r[4] for r in pipeline_results],
        "o-",
        color=C_RFP_BLUE,
        label="RFP  n(n+1)/2 doubles",
    )
    pairs = [(r[0], r[11]) for r in pipeline_results if r[11] is not None]
    if pairs:
        ns_d, mem_d = zip(*pairs)
        ax.plot(ns_d, mem_d, "s--", color=C_PT_RED, label="Dense  n×n doubles")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Matrix memory footprint")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns_all)
    ax.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # ---- [1,0] dlansf ----------------------------------------------------------
    ax = axes[1, 0]
    pairs = _nn(extra_results, 4)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dlansf 1-norm (RFP)")
    pairs = _nn(extra_results, 5)
    if pairs:
        ax.plot(
            *zip(*pairs), "s-", color=C_RFP_BLUE2, label="curfp dlansf Frobenius (RFP)"
        )
    pairs = _nn(extra_results, 6)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch matrix_norm 1 (dense)")
    pairs = _nn(extra_results, 7)
    if pairs:
        ax.plot(
            *zip(*pairs), "D:", color=C_PT_RED2, label="torch matrix_norm fro (dense)"
        )
    style_ax(ax, "dlansf — matrix norm")

    # ---- [1,1] dsfmv ----------------------------------------------------------
    ax = axes[1, 1]
    pairs = _nn(extra_results, 8)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dsfmv (RFP, n(n+1)/2)"
        )
    pairs = _nn(extra_results, 9)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label="cublas dsymv (dense lower tri)",
        )
    pairs = _nn(extra_results, 10)
    if pairs:
        ax.plot(*zip(*pairs), "D:", color=C_PT_RED, label="torch.mv (dense full n×n)")
    style_ax(ax, "dsfmv — symmetric matvec")

    # ---- [1,2] dstrttf ----------------------------------------------------------
    ax = axes[1, 2]
    pairs = _nn(extra_results, 11)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dstrttf (→ RFP n(n+1)/2)"
        )
    pairs = _nn(extra_results, 12)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.triu (→ n×n dense)")
    style_ax(ax, "dstrttf — full tri → packed")

    # ---- [1,3] dstfttr ----------------------------------------------------------
    ax = axes[1, 3]
    pairs = _nn(extra_results, 13)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dstfttr (RFP → full tri)"
        )
    pairs = _nn(extra_results, 14)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.tril (dense → dense)")
    style_ax(ax, "dstfttr — packed → full tri")

    # ---- [1,4] dpfcon ----------------------------------------------------------
    ax = axes[1, 4]
    pairs = _nn(extra_results, 15)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dpfcon O(n²) (RFP)")
    pairs = _nn(extra_results, 16)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.linalg.cond O(n³) SVD")
    style_ax(ax, "dpfcon — condition number")

    # ---- [2,0] dsfr -----------------------------------------------------------
    ax = axes[2, 0]
    pairs = _nn(new_results, 1)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dsfr (RFP n(n+1)/2)")
    pairs = _nn(new_results, 2)
    if pairs:
        ax.plot(
            *zip(*pairs), "s--", color=C_CB_GREEN, label="cublas dsyr (dense lower tri)"
        )
    pairs = _nn(new_results, 3)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.addr (dense full n×n)")
    style_ax(ax, "dsfr — rank-1 update")

    # ---- [2,1] dsfr2 ----------------------------------------------------------
    ax = axes[2, 1]
    pairs = _nn(new_results, 4)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp dsfr2 (RFP)")
    pairs = _nn(new_results, 5)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label="cublas dsyr2 (dense lower tri)",
        )
    pairs = _nn(new_results, 6)
    if pairs:
        ax.plot(
            *zip(*pairs), "^:", color=C_PT_RED, label="torch.addr×2 (dense full n×n)"
        )
    style_ax(ax, "dsfr2 — rank-2 update")

    # ---- [2,2] dsfr2k ---------------------------------------------------------
    ax = axes[2, 2]
    pairs = _nn(new_results, 7)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label=f"curfp dsfr2k (RFP, k={k})"
        )
    pairs = _nn(new_results, 8)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label=f"cublas dsyr2k (dense lower, k={k})",
        )
    pairs = _nn(new_results, 9)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^:",
            color=C_PT_RED,
            label=f"cublas dgemm×2 (dense full, k={k})",
        )
    style_ax(ax, "dsfr2k — rank-2k update")

    # ---- [2,3] dsfmm ----------------------------------------------------------
    ax = axes[2, 3]
    pairs = _nn(new_results, 10)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "o-",
            color=C_RFP_BLUE,
            label=f"curfp dsfmm (RFP, {nrhs_mm} rhs)",
        )
    pairs = _nn(new_results, 11)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label=f"cublas dsymm (dense lower, {nrhs_mm} rhs)",
        )
    pairs = _nn(new_results, 12)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^:",
            color=C_PT_RED,
            label=f"torch.mm (dense full, {nrhs_mm} rhs)",
        )
    style_ax(ax, "dsfmm — symmetric matmul")

    # ---- [2,4] empty ----------------------------------------------------------
    axes[2, 4].set_visible(False)

    plt.tight_layout()
    out = "benchmark64.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")

    # -------------------------------------------------------------------------
    # CSV export
    # -------------------------------------------------------------------------
    csv_out = "benchmark64.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "dsfrk_rfp_ms",
                "dpftrf_rfp_ms",
                "dpftrs_rfp_ms",
                "rfp_mem_gb",
                "dsyrk_dns_ms",
                "dgemm_cb_ms",
                "dgemm_pt_ms",
                "dpotrf_cusolver_ms",
                "cholesky_torch_ms",
                "solve_torch_ms",
                "dense_mem_gb",
                "dpftri_rfp_ms",
                "dpotri_cusolver_ms",
                "inv_torch_ms",
                "dlansf_1_ms",
                "dlansf_F_ms",
                "norm1_torch_ms",
                "normF_torch_ms",
                "dsfmv_rfp_ms",
                "dsymv_cb_ms",
                "mv_torch_ms",
                "dstrttf_rfp_ms",
                "triu_torch_ms",
                "dstfttr_rfp_ms",
                "tril_torch_ms",
                "dpfcon_rfp_ms",
                "cond_torch_ms",
                "dsfr_rfp_ms",
                "dsyr_cb_ms",
                "addr_torch_ms",
                "dsfr2_rfp_ms",
                "dsyr2_cb_ms",
                "addr2_torch_ms",
                "dsfr2k_rfp_ms",
                "dsyr2k_cb_ms",
                "dgemm2_cb_ms",
                "dsfmm_rfp_ms",
                "dsymm_cb_ms",
                "mm_torch_ms",
            ]
        )
        # Build lookup dicts keyed by n
        extra = {r[0]: r for r in extra_results}
        new = {r[0]: r for r in new_results}
        for r in pipeline_results:
            n = r[0]
            e = extra.get(n, [n] + [None] * 16)
            nw = new.get(n, [n] + [None] * 12)
            w.writerow(
                [
                    n,
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                    r[7],
                    r[8],
                    r[9],
                    r[10],
                    r[11],
                    e[1],
                    e[2],
                    e[3],
                    e[4],
                    e[5],
                    e[6],
                    e[7],
                    e[8],
                    e[9],
                    e[10],
                    e[11],
                    e[12],
                    e[13],
                    e[14],
                    e[15],
                    e[16],
                    nw[1],
                    nw[2],
                    nw[3],
                    nw[4],
                    nw[5],
                    nw[6],
                    nw[7],
                    nw[8],
                    nw[9],
                    nw[10],
                    nw[11],
                    nw[12],
                ]
            )
    print(f"Timings saved to {csv_out}")
