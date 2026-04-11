"""
Benchmark: curfp (RFP format) vs PyTorch/cuBLAS/cuSOLVER (dense) for all functions.

curfp functions benchmarked:
  ssfrk  — rank-k update into RFP                  vs cublas ssyrk / torch mm
  spftrf — Cholesky factorization                   vs cusolver spotrf / torch cholesky
  spftrs — triangular solve                         vs torch.cholesky_solve
  spftri — matrix inverse from Cholesky factor      vs cusolver spotri / torch.linalg.inv
  slansf — matrix norm (1-norm and Frobenius)       vs torch.linalg.matrix_norm
  ssfmv  — symmetric matrix-vector product          vs cublas ssymv / cublas spmv / torch.mv
  strttf — full triangular  → RFP conversion        vs cublas strttp (→ TP packed) / torch.triu
  stfttr — RFP → full triangular conversion         vs cublas stpttr (TP packed → full) / torch.tril
  spfcon — condition number estimate (O(n²))        vs torch.linalg.cond (O(n³) SVD)
  ssfr   — rank-1 update in RFP                     vs cublas ssyr / torch.addr
  ssfr2  — rank-2 update in RFP                     vs cublas ssyr2 / torch.addr×2
  ssfr2k — rank-2k update in RFP                    vs cublas ssyr2k / cublas sgemm×2
  ssfmm  — symmetric matrix-matrix multiply in RFP  vs cublas ssymm / cublas sgemm / torch.mm

Colour convention:
  Blue  (#2166ac / #6baed6) — curfp RFP
  Green (#238b45 / #74c476 / #005a32) — cuBLAS / cuSOLVER
  Red   (#cb181d / #fb6a4a) — PyTorch / torch
"""

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

# ---------------------------------------------------------------------------
# cublasSgemm
# ---------------------------------------------------------------------------
_libcublas.cublasSgemm_v2.restype = ctypes.c_int
_libcublas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_sgemm(n, k, alpha, A, lda, beta, C, ldc):
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSgemm_v2(
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
    assert ret == 0, f"cublasSgemm_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSsyrk
# ---------------------------------------------------------------------------
_libcublas.cublasSsyrk_v2.restype = ctypes.c_int
_libcublas.cublasSsyrk_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_ssyrk(n, k, alpha, A, lda, beta, C, ldc):
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSsyrk_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        n,
        k,
        ctypes.byref(_alpha),
        ctypes.c_void_p(A.data_ptr()),
        lda,
        ctypes.byref(_beta),
        ctypes.c_void_p(C.data_ptr()),
        ldc,
    )
    assert ret == 0, f"cublasSsyrk_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSsymv  (symmetric matvec from full n×n lower triangle)
# ---------------------------------------------------------------------------
_libcublas.cublasSsymv_v2.restype = ctypes.c_int
_libcublas.cublasSsymv_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublas_ssymv(n, A, x, y):
    _alpha = ctypes.c_float(1.0)
    _beta = ctypes.c_float(0.0)
    ret = _libcublas.cublasSsymv_v2(
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
    assert ret == 0, f"cublasSsymv_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSspmv  (symmetric matvec from SP packed format, n*(n+1)/2 floats)
# AP is column-major packed: AP[j*(j+1)/2 + i] = A[i,j], i<=j (upper)
# cuBLAS SP format matches the output of cublasSstrttp (TP=SP same layout).
# ---------------------------------------------------------------------------
_libcublas.cublasSspmv_v2.restype = ctypes.c_int
_libcublas.cublasSspmv_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,  # AP: packed n*(n+1)/2 floats
    ctypes.c_void_p,
    ctypes.c_int,  # x, incx
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,  # y, incy
]


def cublas_sspmv(n, AP, x, y):
    """Symmetric matvec from SP packed buffer AP (n*(n+1)/2 floats, upper)."""
    _alpha = ctypes.c_float(1.0)
    _beta = ctypes.c_float(0.0)
    ret = _libcublas.cublasSspmv_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_UPPER,
        n,
        ctypes.byref(_alpha),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        1,
        ctypes.byref(_beta),
        ctypes.c_void_p(y.data_ptr()),
        1,
    )
    assert ret == 0, f"cublasSspmv_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSstrttp  (full triangular → triangular packed / SP format)
# Signature: (handle, uplo, n, A, lda, AP) -> status
# ---------------------------------------------------------------------------
_libcublas.cublasStrttp.restype = ctypes.c_int
_libcublas.cublasStrttp.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,  # A (full n×n), lda
    ctypes.c_void_p,  # AP (packed n*(n+1)/2)
]


def cublas_sstrttp(n, A, AP):
    """Pack upper triangle of full n×n matrix A into TP/SP packed format AP."""
    ret = _libcublas.cublasStrttp(
        _cublas_handle,
        CUBLAS_FILL_MODE_UPPER,
        n,
        ctypes.c_void_p(A.data_ptr()),
        n,
        ctypes.c_void_p(AP.data_ptr()),
    )
    assert ret == 0, f"cublasStrttp returned {ret}"


# ---------------------------------------------------------------------------
# cublasSstpttr  (triangular packed → full triangular)
# Signature: (handle, uplo, n, AP, A, lda) -> status
# ---------------------------------------------------------------------------
_libcublas.cublasStpttr.restype = ctypes.c_int
_libcublas.cublasStpttr.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,  # AP (packed n*(n+1)/2)
    ctypes.c_void_p,
    ctypes.c_int,  # A (full n×n output), lda
]


def cublas_sstpttr(n, AP, A):
    """Unpack TP/SP packed format AP into upper triangle of full n×n matrix A."""
    ret = _libcublas.cublasStpttr(
        _cublas_handle,
        CUBLAS_FILL_MODE_UPPER,
        n,
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(A.data_ptr()),
        n,
    )
    assert ret == 0, f"cublasStpttr returned {ret}"


# ---------------------------------------------------------------------------
# cublasSsyr  (symmetric rank-1 update on dense n×n lower triangle)
# Signature: ssyr(handle, uplo, n, alpha, x, incx, A, lda) -> status
# ---------------------------------------------------------------------------
_libcublas.cublasSsyr_v2.restype = ctypes.c_int
_libcublas.cublasSsyr_v2.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,  # uplo
    ctypes.c_int,  # n
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.c_void_p,  # x
    ctypes.c_int,  # incx
    ctypes.c_void_p,  # A
    ctypes.c_int,  # lda
]


def cublas_ssyr(n, x, A):
    """Dense symmetric rank-1 update: A += alpha * x * x^T (lower tri)."""
    _alpha = ctypes.c_float(1.0)
    ret = _libcublas.cublasSsyr_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        ctypes.byref(_alpha),
        ctypes.c_void_p(x.data_ptr()),
        1,
        ctypes.c_void_p(A.data_ptr()),
        n,
    )
    assert ret == 0, f"cublasSsyr_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSsyr2  (symmetric rank-2 update on dense n×n lower triangle)
# Signature: ssyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda) -> status
# ---------------------------------------------------------------------------
_libcublas.cublasSsyr2_v2.restype = ctypes.c_int
_libcublas.cublasSsyr2_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,  # uplo
    ctypes.c_int,  # n
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.c_void_p,
    ctypes.c_int,  # x, incx
    ctypes.c_void_p,
    ctypes.c_int,  # y, incy
    ctypes.c_void_p,
    ctypes.c_int,  # A, lda
]


def cublas_ssyr2(n, x, y, A):
    """Dense symmetric rank-2 update: A += alpha*(x*y^T + y*x^T) (lower tri)."""
    _alpha = ctypes.c_float(1.0)
    ret = _libcublas.cublasSsyr2_v2(
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
    assert ret == 0, f"cublasSsyr2_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSsyr2k  (symmetric rank-2k update on dense n×n lower triangle)
# Signature: ssyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
# ---------------------------------------------------------------------------
_libcublas.cublasSsyr2k_v2.restype = ctypes.c_int
_libcublas.cublasSsyr2k_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,  # uplo
    ctypes.c_int,  # trans
    ctypes.c_int,  # n
    ctypes.c_int,  # k
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.c_void_p,
    ctypes.c_int,  # A, lda
    ctypes.c_void_p,
    ctypes.c_int,  # B, ldb
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.c_void_p,
    ctypes.c_int,  # C, ldc
]


def cublas_ssyr2k(n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Dense symmetric rank-2k update: C = alpha*(A*B^T + B*A^T) + beta*C (lower tri)."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSsyr2k_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,  # trans=N: A and B are n×k col-major
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
    assert ret == 0, f"cublasSsyr2k_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cublasSsymm  (symmetric matrix-matrix multiply: C = alpha*A*B + beta*C)
# Signature: ssymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
# ---------------------------------------------------------------------------
CUBLAS_SIDE_LEFT = 0
CUBLAS_SIDE_RIGHT = 1

_libcublas.cublasSsymm_v2.restype = ctypes.c_int
_libcublas.cublasSsymm_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,  # side
    ctypes.c_int,  # uplo
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.c_void_p,
    ctypes.c_int,  # A(m×m or n×n), lda
    ctypes.c_void_p,
    ctypes.c_int,  # B(m×n), ldb
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.c_void_p,
    ctypes.c_int,  # C(m×n), ldc
]


def cublas_ssymm(m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Dense C(m×n) = alpha*A(m×m)*B(m×n) + beta*C (side=LEFT, lower tri, col-major)."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSsymm_v2(
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
    assert ret == 0, f"cublasSsymm_v2 returned {ret}"


# ---------------------------------------------------------------------------
# cuSOLVER handle + spotrf / spotri
# ---------------------------------------------------------------------------
_libcusolver.cusolverDnCreate.restype = ctypes.c_int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcusolver.cusolverDnDestroy.restype = ctypes.c_int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverDnSpotrf_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrf_bufferSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_libcusolver.cusolverDnSpotrf.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrf.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]

_libcusolver.cusolverDnSpotri_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnSpotri_bufferSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_libcusolver.cusolverDnSpotri.restype = ctypes.c_int
_libcusolver.cusolverDnSpotri.argtypes = [
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


def cusolver_spotrf(n, S):
    if n not in _cusolver_workspace:
        lwork = ctypes.c_int(0)
        ret = _libcusolver.cusolverDnSpotrf_bufferSize(
            _cusolver_handle,
            CUSOLVER_FILL_MODE_LOWER,
            n,
            ctypes.c_void_p(S.data_ptr()),
            n,
            ctypes.byref(lwork),
        )
        if ret != 0:
            raise RuntimeError(f"cusolverDn bufferSize returned {ret}")
        workspace = torch.empty(lwork.value, dtype=torch.float32, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")
        _cusolver_workspace[n] = (workspace, devInfo, lwork.value)
    workspace, devInfo, lwork = _cusolver_workspace[n]
    ret = _libcusolver.cusolverDnSpotrf(
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
        raise RuntimeError(f"cusolverDnSpotrf returned {ret}")
    return devInfo.item()


def cusolver_spotri(n, S):
    if n not in _cusolver_potri_workspace:
        lwork = ctypes.c_int(0)
        ret = _libcusolver.cusolverDnSpotri_bufferSize(
            _cusolver_handle,
            CUSOLVER_FILL_MODE_LOWER,
            n,
            ctypes.c_void_p(S.data_ptr()),
            n,
            ctypes.byref(lwork),
        )
        if ret != 0:
            raise RuntimeError(f"cusolverDn bufferSize returned {ret}")
        workspace = torch.empty(lwork.value, dtype=torch.float32, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")
        _cusolver_potri_workspace[n] = (workspace, devInfo, lwork.value)
    workspace, devInfo, lwork = _cusolver_potri_workspace[n]
    ret = _libcusolver.cusolverDnSpotri(
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
        raise RuntimeError(f"cusolverDnSpotri returned {ret}")
    return devInfo.item()


# ---------------------------------------------------------------------------
# Timing helper
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
# Pipeline benchmarks: ssfrk / spftrf / spftrs
# ---------------------------------------------------------------------------


def bench_curfp(n, k, nrhs):
    mem_gb = n * (n + 1) // 2 * 4 / 1024**3
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None, mem_gb

    t_sfrk, _ = cuda_time(lambda: None, lambda: curfp.ssfrk(A, C))

    curfp.add_to_diagonal(C, 1.0)
    t_chol, _ = cuda_time(
        lambda: (curfp.ssfrk(A, C), curfp.add_to_diagonal(C, 1.0)),
        lambda: curfp.spftrf(C, check=False),
    )
    t_solve, _ = cuda_time(
        lambda: (
            curfp.ssfrk(A, C),
            curfp.add_to_diagonal(C, 1.0),
            curfp.spftrf(C, check=False),
        ),
        lambda: curfp.spftrs(C, B),
    )

    mem_gb = C.numel() * C.element_size() / 1024**3
    del A, C, B
    return t_sfrk, t_chol, t_solve, mem_gb


def bench_torch(n, k, nrhs):
    mat_gb = n * n * 4 / 1024**3
    t_torch_sgemm = t_cublas_sgemm = t_ssyrk = None
    t_torch_chol = t_cusolver_chol = t_solve = None

    try:
        A = (torch.randn(k, n, dtype=torch.float32, device="cuda") / k**0.5).t()
        S = torch.empty(n, n, dtype=torch.float32, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return (
            t_torch_sgemm,
            t_cublas_sgemm,
            t_ssyrk,
            t_torch_chol,
            t_cusolver_chol,
            t_solve,
            mat_gb,
        )

    t_torch_sgemm, _ = cuda_time(lambda: None, lambda: torch.mm(A, A.t(), out=S))
    t_cublas_sgemm, _ = cuda_time(
        lambda: None, lambda: cublas_sgemm(n, k, 1.0, A, n, 0.0, S, n)
    )
    t_ssyrk, _ = cuda_time(
        lambda: None, lambda: cublas_ssyrk(n, k, 1.0, A, n, 0.0, S, n)
    )

    def ssyrk_diag():
        cublas_ssyrk(n, k, 1.0, A, n, 0.0, S, n)
        S.diagonal().add_(1.0)

    def sgemm_diag():
        torch.mm(A, A.t(), out=S)
        S.diagonal().add_(1.0)

    try:
        t_cusolver_chol, info = cuda_time(ssyrk_diag, lambda: cusolver_spotrf(n, S))
        assert info == 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    try:
        t_torch_chol, _ = cuda_time(sgemm_diag, lambda: torch.linalg.cholesky(S, out=S))
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
        t_torch_sgemm,
        t_cublas_sgemm,
        t_ssyrk,
        t_torch_chol,
        t_cusolver_chol,
        t_solve,
        mat_gb,
    )


# ---------------------------------------------------------------------------
# spftri
# ---------------------------------------------------------------------------


def bench_spftri(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")

        def setup():
            curfp.ssfrk(A, C)
            curfp.add_to_diagonal(C, 1.0)
            curfp.spftrf(C, check=False)

        t_rfp, _ = cuda_time(setup, lambda: curfp.spftri(C))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_potri = t_inv = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.zeros(n, n, dtype=torch.float32, device="cuda")

        def setup_chol():
            torch.mm(A2, A2.t(), out=S)
            S.diagonal().add_(1.0)
            cusolver_spotrf(n, S)

        setup_chol()
        t_potri, _ = cuda_time(setup_chol, lambda: cusolver_spotri(n, S))

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
# slansf
# ---------------------------------------------------------------------------


def bench_slansf(n, k):
    torch.cuda.empty_cache()
    t_1 = t_fro = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_1, _ = cuda_time(lambda: None, lambda: curfp.slansf(C, "1"))
        t_fro, _ = cuda_time(lambda: None, lambda: curfp.slansf(C, "F"))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_d1 = t_dfro = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
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
# ssfmv  — adds cublasSspmv (packed SP format, same footprint as RFP)
# ---------------------------------------------------------------------------

# Cache for pre-converted SP packed buffers (one per n)
_spmv_sp_cache = {}


def bench_ssfmv(n, k):
    """ssfmv (RFP) vs cublasSsymv (full n×n lower tri)
    vs cublasSspmv (SP packed n*(n+1)/2, same footprint as RFP)
    vs torch.mv   (full n×n)"""
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        x = torch.randn(n, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.ssfmv(C, x))
        del A, C, x
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_symv = t_spmv = t_mv = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float32, device="cuda")
        y_symv = torch.empty(n, dtype=torch.float32, device="cuda")
        y_spmv = torch.empty(n, dtype=torch.float32, device="cuda")

        # cublasSsymv: reads only lower triangle of full n×n S
        t_symv, _ = cuda_time(lambda: None, lambda: cublas_ssymv(n, S, x2, y_symv))

        # cublasSspmv: operates on SP packed format (n*(n+1)/2 floats, upper).
        # Pre-pack S into SP format once outside the timed loop — not counted.
        # Dedicated output buffer y_spmv — no aliasing with other timed calls.
        if n not in _spmv_sp_cache:
            nt = n * (n + 1) // 2
            AP = torch.empty(nt, dtype=torch.float32, device="cuda")
            cublas_sstrttp(n, S, AP)
            torch.cuda.synchronize()
            _spmv_sp_cache[n] = AP
        AP = _spmv_sp_cache[n]
        t_spmv, _ = cuda_time(lambda: None, lambda: cublas_sspmv(n, AP, x2, y_spmv))

        # torch.mv: full general matvec on n×n S
        t_mv, _ = cuda_time(lambda: None, lambda: torch.mv(S, x2))

        del A2, S, x2, y_symv, y_spmv
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_symv, t_spmv, t_mv


# ---------------------------------------------------------------------------
# strttf  — adds cublasSstrttp (full → TP packed, same n*(n+1)/2 footprint)
# ---------------------------------------------------------------------------


def bench_strttf(n, k):
    """strttf (full → RFP) vs cublasSstrttp (full → TP packed) vs torch.triu (full → full)."""
    torch.cuda.empty_cache()
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.mm(A, A.t())
        S.diagonal().add_(1.0)
        tri = torch.triu(S)
        del S, A
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None, None, None

    t_rfp = t_cb = t_dense = None
    try:
        # curfp strttf: full upper tri → RFP (n*(n+1)/2)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.strttf(tri, uplo="U"))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    try:
        # cublasSstrttp: full upper tri → TP packed (n*(n+1)/2)
        nt = n * (n + 1) // 2
        AP = torch.empty(nt, dtype=torch.float32, device="cuda")
        t_cb, _ = cuda_time(lambda: None, lambda: cublas_sstrttp(n, tri, AP))
        del AP
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    try:
        # torch.triu: full → full (2× memory output, kept as reference)
        t_dense, _ = cuda_time(lambda: None, lambda: torch.triu(tri))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass

    del tri
    torch.cuda.empty_cache()
    return t_rfp, t_cb, t_dense


# ---------------------------------------------------------------------------
# stfttr  — adds cublasSstpttr (TP packed → full)
# ---------------------------------------------------------------------------


def bench_stfttr(n, k):
    """stfttr (RFP → full) vs cublasSstpttr (TP packed → full) vs torch.tril (full → full)."""
    torch.cuda.empty_cache()
    t_rfp = t_cb = t_dense = None

    # --- curfp stfttr ---
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.stfttr(C, uplo="U"))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    # --- cublasSstpttr and torch.tril ---
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        out = torch.empty(n, n, dtype=torch.float32, device="cuda")

        # Pre-pack S into TP format for stpttr benchmark
        nt = n * (n + 1) // 2
        AP = torch.empty(nt, dtype=torch.float32, device="cuda")
        cublas_sstrttp(n, S, AP)
        torch.cuda.synchronize()

        # cublasSstpttr: TP packed → full upper triangle
        t_cb, _ = cuda_time(lambda: None, lambda: cublas_sstpttr(n, AP, out))

        # torch.tril on full n×n (reference: same element count written)
        t_dense, _ = cuda_time(lambda: None, lambda: torch.tril(S))

        del A2, S, AP, out
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_cb, t_dense


# ---------------------------------------------------------------------------
# ssfr  — rank-1 update
# ---------------------------------------------------------------------------


def bench_ssfr(n, k):
    """ssfr (RFP) vs cublasSsyr (full dense lower tri) vs torch.addr (full dense)."""
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        x = torch.randn(n, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.ssfr(C, x))
        del A, C, x
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_ssyr = t_addr = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float32, device="cuda")
        # cublasSsyr: in-place rank-1 update on lower triangle of full n×n
        t_ssyr, _ = cuda_time(lambda: None, lambda: cublas_ssyr(n, x2, S))
        # torch.addr: full rank-1 update (both triangles, 2× writes)
        t_addr, _ = cuda_time(lambda: None, lambda: torch.addr(S, x2, x2, out=S))
        del A2, S, x2
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_ssyr, t_addr


# ---------------------------------------------------------------------------
# ssfr2  — rank-2 update
# ---------------------------------------------------------------------------


def bench_ssfr2(n, k):
    """ssfr2 (RFP) vs cublasSsyr2 (dense lower tri) vs torch.addr×2 (dense full)."""
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        x = torch.randn(n, dtype=torch.float32, device="cuda")
        y = torch.randn(n, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.ssfr2(C, x, y))
        del A, C, x, y
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_ssyr2 = t_addr2 = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float32, device="cuda")
        y2 = torch.randn(n, dtype=torch.float32, device="cuda")
        t_ssyr2, _ = cuda_time(lambda: None, lambda: cublas_ssyr2(n, x2, y2, S))
        # torch.addr twice (x*y^T + y*x^T) on full matrix
        t_addr2, _ = cuda_time(
            lambda: None,
            lambda: (torch.addr(S, x2, y2, out=S), torch.addr(S, y2, x2, out=S)),
        )
        del A2, S, x2, y2
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_ssyr2, t_addr2


# ---------------------------------------------------------------------------
# ssfr2k  — rank-2k update
# ---------------------------------------------------------------------------


def bench_ssfr2k(n, k):
    """ssfr2k (RFP) vs cublasSsyr2k (dense lower tri) vs cublasSgemm×2 (dense full)."""
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        # A and B are (n, k) row-major for trans='T' convention
        A_rfp = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        B_rfp = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C_rfp = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A_rfp, C_rfp)
        curfp.add_to_diagonal(C_rfp, 1.0)
        t_rfp, _ = cuda_time(
            lambda: None,
            lambda: curfp.ssfr2k(A_rfp, B_rfp, C_rfp, beta=1.0),
        )
        del A_rfp, B_rfp, C_rfp
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_ssyr2k = t_gemm2 = None
    try:
        # Dense references use col-major convention: A_dns is (k,n) row-major → n×k col-major
        # For ssyr2k(NOTRANS, n, k): A and B are n×k col-major with lda=n
        # Use random (n,k) tensors with lda=n (Fortran-order would be cleaner but
        # lda=n for NOTRANS matches cuBLAS convention with C-contiguous (n,k) tensors)
        A_dns = torch.randn(k, n, dtype=torch.float32, device="cuda").t().contiguous()
        B_dns = torch.randn(k, n, dtype=torch.float32, device="cuda").t().contiguous()
        S = torch.zeros(n, n, dtype=torch.float32, device="cuda")
        torch.mm(A_dns, A_dns.t(), out=S)
        S.diagonal().add_(1.0)
        # cublasSsyr2k(NOTRANS, n, k): A is n×k col-major lda=n, C is n×n lower
        t_ssyr2k, _ = cuda_time(
            lambda: None,
            lambda: cublas_ssyr2k(n, k, 1.0, A_dns, n, B_dns, n, 1.0, S, n),
        )

        # cublasSgemm×2: A*B^T + B*A^T on full n×n (NOTRANS×TRANS pair)
        def two_gemm():
            cublas_sgemm(n, k, 1.0, A_dns, n, 0.0, S, n)
            cublas_sgemm(n, k, 1.0, B_dns, n, 1.0, S, n)

        t_gemm2, _ = cuda_time(lambda: None, two_gemm)
        del A_dns, B_dns, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_ssyr2k, t_gemm2


# ---------------------------------------------------------------------------
# ssfmm  — symmetric matrix-matrix multiply
# ---------------------------------------------------------------------------


def bench_ssfmm(n, k, nrhs):
    """ssfmm (RFP) vs cublasSsymm (dense sym, lower tri) vs cublasSgemm (dense full)
    vs torch.mm (full dense)."""
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A_rfp = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C_rfp = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        # B and out as (nrhs, n) row-major; ssfmm convention: m=n, n_p=nrhs, ldb=n
        B_rfp = torch.randn(nrhs, n, dtype=torch.float32, device="cuda")
        out_rfp = torch.empty(nrhs, n, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A_rfp, C_rfp)
        curfp.add_to_diagonal(C_rfp, 1.0)
        h = curfp.Handle()
        t_rfp, _ = cuda_time(
            lambda: None,
            lambda: curfp.ssfmm_raw(
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

    t_ssymm = t_mm = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        # B and C are n×nrhs col-major for cuBLAS ssymm (side=LEFT, m=n, n=nrhs)
        B_dns = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
        C_dns = torch.empty(n, nrhs, dtype=torch.float32, device="cuda")
        # cublasSsymm(LEFT, LOWER, n, nrhs): C = S*B (n×n * n×nrhs = n×nrhs col-major)
        t_ssymm, _ = cuda_time(
            lambda: None,
            lambda: cublas_ssymm(n, nrhs, 1.0, S, n, B_dns, n, 0.0, C_dns, n),
        )
        # torch.mm: full dense matmul (no symmetry exploit, general n×n × n×nrhs)
        t_mm, _ = cuda_time(lambda: None, lambda: torch.mm(S, B_dns))
        del A2, S, B_dns, C_dns
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_ssymm, t_mm


# ---------------------------------------------------------------------------
# spfcon
# ---------------------------------------------------------------------------


def bench_spfcon(n, k):
    torch.cuda.empty_cache()
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        anorm = curfp.slansf(C, "1")
        curfp.spftrf(C, check=False)
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.spfcon(C, anorm))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dense = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
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
#   Blue  = curfp RFP
#   Green = cuBLAS / cuSOLVER
#   Red   = PyTorch / torch
# ---------------------------------------------------------------------------
C_RFP_BLUE = "#2166ac"  # curfp primary
C_RFP_BLUE2 = "#6baed6"  # curfp secondary (e.g. Frobenius)
C_CB_GREEN = "#238b45"  # cuBLAS primary
C_CB_GREEN2 = "#74c476"  # cuBLAS secondary
C_CS_GREEN = "#005a32"  # cuSOLVER (darker green)
C_PT_RED = "#cb181d"  # torch primary
C_PT_RED2 = "#fb6a4a"  # torch secondary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 56000, 65536, 80000]
    # sizes = [1024, 2048, 4096, 8192, 16384]
    k = 1024
    nrhs = 64
    nrhs_mm = 1024  # wider nrhs for ssfmm to show matmul throughput advantage

    print(
        f"Benchmark: curfp (RFP) vs dense — all ops, float32, rank-{k} update, {nrhs} rhs"
    )

    # ---- Table 1: pipeline ---------------------------------------------------
    print()
    print(
        f"{'n':>6}  {'curfp ssfrk':>12} {'curfp spftrf':>13} {'curfp spftrs':>13} {'curfp mem':>10}"
        f"  {'ssyrk':>8} {'cb sgemm':>9} {'pt sgemm':>9} {'cusol chol':>11} {'torch chol':>11} {'torch solve':>12} {'dense mem':>10}"
    )
    print("-" * 170)

    pipeline_results = []

    for n in sizes:
        t_sfrk, t_chol_rfp, t_solve_rfp, mem_rfp = bench_curfp(n, k, nrhs)
        torch.cuda.empty_cache()
        (
            t_pt_sgemm,
            t_cb_sgemm,
            t_ssyrk,
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
                t_ssyrk,
                t_cb_sgemm,
                t_pt_sgemm,
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
            f"  {_ms1(t_ssyrk)}"
            f"  {_ms1(t_cb_sgemm)}"
            f"  {_ms1(t_pt_sgemm)}"
            f"  {_ms1(t_cusolver_chol)}"
            f"  {_ms1(t_torch_chol)}"
            f"  {_ms1(t_solve_dns)}"
            f"  {mem_dns:>8.3f} GB"
        )

    # ---- Table 2: remaining ops ----------------------------------------------
    print()
    print(
        f"{'n':>6}  {'spftri rfp':>11} {'potri dns':>10} {'inv dns':>9}"
        f"  {'slansf 1':>9} {'slansf F':>9} {'norm1 dns':>10} {'normF dns':>10}"
        f"  {'ssfmv rfp':>10} {'symv dns':>9} {'spmv dns':>9} {'mv dense':>9}"
        f"  {'strttf':>8} {'trttp cb':>9} {'triu dns':>9}"
        f"  {'stfttr':>8} {'stpttr cb':>10} {'tril dns':>9}"
        f"  {'spfcon':>8} {'cond dns':>9}"
    )
    print("-" * 230)

    extra_results = []

    for n in sizes:
        t_spftri_r, t_potri_d, t_inv_d = bench_spftri(n, k)
        torch.cuda.empty_cache()
        t_s1, t_sf, t_d1, t_df = bench_slansf(n, k)
        torch.cuda.empty_cache()
        t_ssfmv_r, t_symv_d, t_spmv_d, t_mv_d = bench_ssfmv(n, k)
        torch.cuda.empty_cache()
        t_strttf_r, t_strttf_cb, t_strttf_d = bench_strttf(n, k)
        torch.cuda.empty_cache()
        t_stfttr_r, t_stfttr_cb, t_stfttr_d = bench_stfttr(n, k)
        torch.cuda.empty_cache()
        t_spfcon_r, t_spfcon_d = bench_spfcon(n, k)
        torch.cuda.empty_cache()

        # Index map:
        #  0:n
        #  1:spftri_r   2:potri_d    3:inv_d
        #  4:s1         5:sf         6:d1         7:df
        #  8:ssfmv_r    9:symv_d    10:spmv_d    11:mv_d
        # 12:strttf_r  13:strttf_cb 14:strttf_d
        # 15:stfttr_r  16:stfttr_cb 17:stfttr_d
        # 18:spfcon_r  19:spfcon_d
        extra_results.append(
            (
                n,
                t_spftri_r,
                t_potri_d,
                t_inv_d,
                t_s1,
                t_sf,
                t_d1,
                t_df,
                t_ssfmv_r,
                t_symv_d,
                t_spmv_d,
                t_mv_d,
                t_strttf_r,
                t_strttf_cb,
                t_strttf_d,
                t_stfttr_r,
                t_stfttr_cb,
                t_stfttr_d,
                t_spfcon_r,
                t_spfcon_d,
            )
        )

        print(
            f"{n:>6}"
            f"  {_ms2(t_spftri_r)} {_ms2(t_potri_d)} {_ms2(t_inv_d)}"
            f"  {_ms2(t_s1)} {_ms2(t_sf)} {_ms2(t_d1)} {_ms2(t_df)}"
            f"  {_ms2(t_ssfmv_r)} {_ms2(t_symv_d)} {_ms2(t_spmv_d)} {_ms2(t_mv_d)}"
            f"  {_ms2(t_strttf_r)} {_ms2(t_strttf_cb)} {_ms2(t_strttf_d)}"
            f"  {_ms2(t_stfttr_r)} {_ms2(t_stfttr_cb)} {_ms2(t_stfttr_d)}"
            f"  {_ms2(t_spfcon_r)} {_ms2(t_spfcon_d)}"
        )

    # ---- Table 3: new RFP ops (ssfr, ssfr2, ssfr2k, ssfmm) ------------------
    print()
    print(
        f"{'n':>6}"
        f"  {'ssfr rfp':>9} {'ssyr dns':>9} {'addr dns':>9}"
        f"  {'ssfr2 rfp':>10} {'ssyr2 dns':>10} {'addr2 dns':>10}"
        f"  {'ssfr2k rfp':>11} {'ssyr2k dns':>11} {'gemm2 dns':>10}"
        f"  {'ssfmm rfp':>10} {'ssymm dns':>10} {'mm dns':>8}  (ssfmm nrhs={nrhs_mm})"
    )
    print("-" * 175)

    new_results = []

    for n in sizes:
        t_ssfr_r, t_ssyr_d, t_addr_d = bench_ssfr(n, k)
        torch.cuda.empty_cache()
        t_ssfr2_r, t_ssyr2_d, t_addr2_d = bench_ssfr2(n, k)
        torch.cuda.empty_cache()
        t_ssfr2k_r, t_ssyr2k_d, t_gemm2_d = bench_ssfr2k(n, k)
        torch.cuda.empty_cache()
        t_ssfmm_r, t_ssymm_d, t_mm_d = bench_ssfmm(n, k, nrhs_mm)
        torch.cuda.empty_cache()

        # Index map:
        #  0:n
        #  1:ssfr_r    2:ssyr_d    3:addr_d
        #  4:ssfr2_r   5:ssyr2_d   6:addr2_d
        #  7:ssfr2k_r  8:ssyr2k_d  9:gemm2_d
        # 10:ssfmm_r  11:ssymm_d  12:mm_d
        new_results.append(
            (
                n,
                t_ssfr_r,
                t_ssyr_d,
                t_addr_d,
                t_ssfr2_r,
                t_ssyr2_d,
                t_addr2_d,
                t_ssfr2k_r,
                t_ssyr2k_d,
                t_gemm2_d,
                t_ssfmm_r,
                t_ssymm_d,
                t_mm_d,
            )
        )

        print(
            f"{n:>6}"
            f"  {_ms2(t_ssfr_r)} {_ms2(t_ssyr_d)} {_ms2(t_addr_d)}"
            f"  {_ms2(t_ssfr2_r)} {_ms2(t_ssyr2_d)} {_ms2(t_addr2_d)}"
            f"  {_ms2(t_ssfr2k_r)} {_ms2(t_ssyr2k_d)} {_ms2(t_gemm2_d)}"
            f"  {_ms2(t_ssfmm_r)} {_ms2(t_ssymm_d)} {_ms2(t_mm_d)}"
        )

    # -------------------------------------------------------------------------
    # Plot: 3 rows × 5 columns
    # Row 0-1: existing ops  Row 2: new RFP ops (ssfr, ssfr2, ssfr2k, ssfmm)
    # Blue = curfp RFP, Green = cuBLAS/cuSOLVER, Red = PyTorch/torch
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(3, 5, figsize=(30, 18))
    fig.suptitle(
        f"curfp (RFP) vs dense — all ops, float32, rank-{k} update, {nrhs} rhs",
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

    # ---- [0,0] ssfrk ----------------------------------------------------------
    ax = axes[0, 0]
    pairs = _nn(pipeline_results, 1)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp ssfrk (RFP)")
    pairs = _nn(pipeline_results, 5)
    if pairs:
        ax.plot(
            *zip(*pairs), "s--", color=C_CB_GREEN, label="cublas ssyrk (dense lower)"
        )
    pairs = _nn(pipeline_results, 6)
    if pairs:
        ax.plot(
            *zip(*pairs), "^--", color=C_CB_GREEN2, label="cublas sgemm (dense full)"
        )
    pairs = _nn(pipeline_results, 7)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch mm (dense full)")
    style_ax(ax, "ssfrk — rank-k update")

    # ---- [0,1] spftrf ----------------------------------------------------------
    ax = axes[0, 1]
    pairs = _nn(pipeline_results, 2)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp spftrf (RFP)")
    pairs = _nn(pipeline_results, 8)
    if pairs:
        ax.plot(*zip(*pairs), "D-.", color=C_CS_GREEN, label="cusolver spotrf (dense)")
    pairs = _nn(pipeline_results, 9)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch cholesky (dense)")
    style_ax(ax, "spftrf — Cholesky factorization")

    # ---- [0,2] spftrs ----------------------------------------------------------
    ax = axes[0, 2]
    pairs = _nn(pipeline_results, 3)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "o-",
            color=C_RFP_BLUE,
            label=f"curfp spftrs (RFP, {nrhs} rhs)",
        )
    pairs = _nn(pipeline_results, 10)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^:",
            color=C_PT_RED,
            label=f"torch.cholesky_solve ({nrhs} rhs)",
        )
    style_ax(ax, "spftrs — triangular solve")

    # ---- [0,3] spftri ----------------------------------------------------------
    ax = axes[0, 3]
    pairs = _nn(extra_results, 1)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp spftri (RFP)")
    pairs = _nn(extra_results, 2)
    if pairs:
        ax.plot(*zip(*pairs), "D-.", color=C_CS_GREEN, label="cusolver spotri (dense)")
    pairs = _nn(extra_results, 3)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.linalg.inv (dense LU)")
    style_ax(ax, "spftri — matrix inversion")

    # ---- [0,4] memory ----------------------------------------------------------
    ax = axes[0, 4]
    ax.plot(
        ns_all,
        [r[4] for r in pipeline_results],
        "o-",
        color=C_RFP_BLUE,
        label="RFP  n(n+1)/2 floats",
    )
    pairs = [(r[0], r[11]) for r in pipeline_results if r[11] is not None]
    if pairs:
        ns_d, mem_d = zip(*pairs)
        ax.plot(ns_d, mem_d, "s--", color=C_PT_RED, label="Dense  n×n floats")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Matrix memory footprint")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns_all)
    ax.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # ---- [1,0] slansf ----------------------------------------------------------
    ax = axes[1, 0]
    pairs = _nn(extra_results, 4)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp slansf 1-norm (RFP)")
    pairs = _nn(extra_results, 5)
    if pairs:
        ax.plot(
            *zip(*pairs), "s-", color=C_RFP_BLUE2, label="curfp slansf Frobenius (RFP)"
        )
    pairs = _nn(extra_results, 6)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch matrix_norm 1 (dense)")
    pairs = _nn(extra_results, 7)
    if pairs:
        ax.plot(
            *zip(*pairs), "D:", color=C_PT_RED2, label="torch matrix_norm fro (dense)"
        )
    style_ax(ax, "slansf — matrix norm")

    # ---- [1,1] ssfmv ----------------------------------------------------------
    ax = axes[1, 1]
    pairs = _nn(extra_results, 8)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp ssfmv (RFP, n(n+1)/2)"
        )
    pairs = _nn(extra_results, 9)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label="cublas ssymv (dense lower tri)",
        )
    pairs = _nn(extra_results, 10)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^--",
            color=C_CB_GREEN2,
            label="cublas spmv (SP packed, n(n+1)/2)",
        )
    pairs = _nn(extra_results, 11)
    if pairs:
        ax.plot(*zip(*pairs), "D:", color=C_PT_RED, label="torch.mv (dense full n×n)")
    style_ax(ax, "ssfmv — symmetric matvec")

    # ---- [1,2] strttf ----------------------------------------------------------
    ax = axes[1, 2]
    pairs = _nn(extra_results, 12)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp strttf (→ RFP n(n+1)/2)"
        )
    pairs = _nn(extra_results, 13)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label="cublas strttp (→ TP packed n(n+1)/2)",
        )
    pairs = _nn(extra_results, 14)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.triu (→ n×n dense)")
    style_ax(ax, "strttf — full tri → packed")

    # ---- [1,3] stfttr ----------------------------------------------------------
    ax = axes[1, 3]
    pairs = _nn(extra_results, 15)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp stfttr (RFP → full tri)"
        )
    pairs = _nn(extra_results, 16)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label="cublas stpttr (TP packed → full)",
        )
    pairs = _nn(extra_results, 17)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.tril (dense → dense)")
    style_ax(ax, "stfttr — packed → full tri")

    # ---- [1,4] spfcon ----------------------------------------------------------
    ax = axes[1, 4]
    pairs = _nn(extra_results, 18)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp spfcon O(n²) (RFP)")
    pairs = _nn(extra_results, 19)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.linalg.cond O(n³) SVD")
    style_ax(ax, "spfcon — condition number")

    # ---- [2,0] ssfr -----------------------------------------------------------
    ax = axes[2, 0]
    pairs = _nn(new_results, 1)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp ssfr (RFP n(n+1)/2)")
    pairs = _nn(new_results, 2)
    if pairs:
        ax.plot(
            *zip(*pairs), "s--", color=C_CB_GREEN, label="cublas ssyr (dense lower tri)"
        )
    pairs = _nn(new_results, 3)
    if pairs:
        ax.plot(*zip(*pairs), "^:", color=C_PT_RED, label="torch.addr (dense full n×n)")
    style_ax(ax, "ssfr — rank-1 update")

    # ---- [2,1] ssfr2 ----------------------------------------------------------
    ax = axes[2, 1]
    pairs = _nn(new_results, 4)
    if pairs:
        ax.plot(*zip(*pairs), "o-", color=C_RFP_BLUE, label="curfp ssfr2 (RFP)")
    pairs = _nn(new_results, 5)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label="cublas ssyr2 (dense lower tri)",
        )
    pairs = _nn(new_results, 6)
    if pairs:
        ax.plot(
            *zip(*pairs), "^:", color=C_PT_RED, label="torch.addr×2 (dense full n×n)"
        )
    style_ax(ax, "ssfr2 — rank-2 update")

    # ---- [2,2] ssfr2k ---------------------------------------------------------
    ax = axes[2, 2]
    pairs = _nn(new_results, 7)
    if pairs:
        ax.plot(
            *zip(*pairs), "o-", color=C_RFP_BLUE, label=f"curfp ssfr2k (RFP, k={k})"
        )
    pairs = _nn(new_results, 8)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label=f"cublas ssyr2k (dense lower, k={k})",
        )
    pairs = _nn(new_results, 9)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^:",
            color=C_PT_RED,
            label=f"cublas sgemm×2 (dense full, k={k})",
        )
    style_ax(ax, "ssfr2k — rank-2k update")

    # ---- [2,3] ssfmm ----------------------------------------------------------
    ax = axes[2, 3]
    pairs = _nn(new_results, 10)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "o-",
            color=C_RFP_BLUE,
            label=f"curfp ssfmm (RFP, {nrhs_mm} rhs)",
        )
    pairs = _nn(new_results, 11)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "s--",
            color=C_CB_GREEN,
            label=f"cublas ssymm (dense lower, {nrhs_mm} rhs)",
        )
    pairs = _nn(new_results, 12)
    if pairs:
        ax.plot(
            *zip(*pairs),
            "^:",
            color=C_PT_RED,
            label=f"torch.mm (dense full, {nrhs_mm} rhs)",
        )
    style_ax(ax, "ssfmm — symmetric matmul")

    # ---- [2,4] empty (placeholder) --------------------------------------------
    axes[2, 4].set_visible(False)

    plt.tight_layout()
    out = "benchmark.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
