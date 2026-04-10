"""
Benchmark: curfp (RFP format) vs PyTorch/cuBLAS/cuSOLVER (dense) for all functions.

curfp functions benchmarked:
  ssfrk  — rank-k update into RFP                  vs cublas ssyrk / torch mm
  spftrf — Cholesky factorization                   vs cusolver spotrf / torch cholesky
  spftrs — triangular solve                         vs torch.cholesky_solve
  spftri — matrix inverse from Cholesky factor      vs torch.linalg.inv
  slansf — matrix norm (1-norm and Frobenius)       vs torch.linalg.matrix_norm
  ssfmv  — symmetric matrix-vector product          vs torch.mv (dense full matrix)
  strttf — full triangular  → RFP conversion        vs torch.triu (stays full n×n)
  stfttr — RFP → full triangular conversion         vs torch.tril (stays full n×n)
  spfcon — condition number estimate (O(n²))        vs torch.linalg.cond (O(n³) SVD)
"""

import ctypes

import torch
import curfp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Minimal ctypes wrapper around cublasSsyrk_v2
# ---------------------------------------------------------------------------
_libcublas = ctypes.CDLL("libcublas.so.12", use_errno=True)

_libcublas.cublasCreate_v2.restype = ctypes.c_int
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcublas.cublasDestroy_v2.restype = ctypes.c_int
_libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

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

_cublas_handle = ctypes.c_void_p()
assert _libcublas.cublasCreate_v2(ctypes.byref(_cublas_handle)) == 0

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

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
        _cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
        ctypes.byref(_alpha), ctypes.c_void_p(A.data_ptr()), lda,
        ctypes.c_void_p(A.data_ptr()), lda,
        ctypes.byref(_beta), ctypes.c_void_p(C.data_ptr()), ldc,
    )
    assert ret == 0, f"cublasSgemm_v2 returned {ret}"


def cublas_ssyrk(n, k, alpha, A, lda, beta, C, ldc):
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSsyrk_v2(
        _cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
        ctypes.byref(_alpha), ctypes.c_void_p(A.data_ptr()), lda,
        ctypes.byref(_beta), ctypes.c_void_p(C.data_ptr()), ldc,
    )
    assert ret == 0, f"cublasSsyrk_v2 returned {ret}"


_libcublas.cublasSsymv_v2.restype = ctypes.c_int
_libcublas.cublasSsymv_v2.argtypes = [
    ctypes.c_void_p,                 # handle
    ctypes.c_int,                    # uplo
    ctypes.c_int,                    # n
    ctypes.POINTER(ctypes.c_float),  # alpha (host)
    ctypes.c_void_p,                 # A (device, n×n)
    ctypes.c_int,                    # lda
    ctypes.c_void_p,                 # x (device, n)
    ctypes.c_int,                    # incx
    ctypes.POINTER(ctypes.c_float),  # beta (host)
    ctypes.c_void_p,                 # y (device, n)
    ctypes.c_int,                    # incy
]


def cublas_ssymv(n, A, x, y):
    _alpha = ctypes.c_float(1.0)
    _beta  = ctypes.c_float(0.0)
    ret = _libcublas.cublasSsymv_v2(
        _cublas_handle, CUBLAS_FILL_MODE_LOWER, n,
        ctypes.byref(_alpha), ctypes.c_void_p(A.data_ptr()), n,
        ctypes.c_void_p(x.data_ptr()), 1,
        ctypes.byref(_beta),  ctypes.c_void_p(y.data_ptr()), 1,
    )
    assert ret == 0, f"cublasSsymv_v2 returned {ret}"


# ---------------------------------------------------------------------------
# Minimal ctypes wrapper around cusolverDnSpotrf
# ---------------------------------------------------------------------------
_libcusolver = ctypes.CDLL("libcusolver.so.11", use_errno=True)

_libcusolver.cusolverDnCreate.restype = ctypes.c_int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcusolver.cusolverDnDestroy.restype = ctypes.c_int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]
_libcusolver.cusolverDnSpotrf_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrf_bufferSize.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
]
_libcusolver.cusolverDnSpotrf.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrf.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
]

CUSOLVER_FILL_MODE_LOWER = 0
_cusolver_handle = ctypes.c_void_p()
assert _libcusolver.cusolverDnCreate(ctypes.byref(_cusolver_handle)) == 0

_libcusolver.cusolverDnSpotri_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnSpotri_bufferSize.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
]
_libcusolver.cusolverDnSpotri.restype = ctypes.c_int
_libcusolver.cusolverDnSpotri.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
]

_cusolver_workspace = {}
_cusolver_potri_workspace = {}


def cusolver_spotrf(n, S):
    if n not in _cusolver_workspace:
        lwork = ctypes.c_int(0)
        ret = _libcusolver.cusolverDnSpotrf_bufferSize(
            _cusolver_handle, CUSOLVER_FILL_MODE_LOWER, n,
            ctypes.c_void_p(S.data_ptr()), n, ctypes.byref(lwork),
        )
        if ret != 0:
            raise RuntimeError(f"cusolverDn bufferSize returned {ret}")
        workspace = torch.empty(lwork.value, dtype=torch.float32, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")
        _cusolver_workspace[n] = (workspace, devInfo, lwork.value)
    workspace, devInfo, lwork = _cusolver_workspace[n]
    ret = _libcusolver.cusolverDnSpotrf(
        _cusolver_handle, CUSOLVER_FILL_MODE_LOWER, n,
        ctypes.c_void_p(S.data_ptr()), n,
        ctypes.c_void_p(workspace.data_ptr()), lwork,
        ctypes.c_void_p(devInfo.data_ptr()),
    )
    if ret != 0:
        raise RuntimeError(f"cusolverDnSpotrf returned {ret}")
    return devInfo.item()


def cusolver_spotri(n, S):
    if n not in _cusolver_potri_workspace:
        lwork = ctypes.c_int(0)
        ret = _libcusolver.cusolverDnSpotri_bufferSize(
            _cusolver_handle, CUSOLVER_FILL_MODE_LOWER, n,
            ctypes.c_void_p(S.data_ptr()), n, ctypes.byref(lwork),
        )
        if ret != 0:
            raise RuntimeError(f"cusolverDn bufferSize returned {ret}")
        workspace = torch.empty(lwork.value, dtype=torch.float32, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")
        _cusolver_potri_workspace[n] = (workspace, devInfo, lwork.value)
    workspace, devInfo, lwork = _cusolver_potri_workspace[n]
    ret = _libcusolver.cusolverDnSpotri(
        _cusolver_handle, CUSOLVER_FILL_MODE_LOWER, n,
        ctypes.c_void_p(S.data_ptr()), n,
        ctypes.c_void_p(workspace.data_ptr()), lwork,
        ctypes.c_void_p(devInfo.data_ptr()),
    )
    if ret != 0:
        raise RuntimeError(f"cusolverDnSpotri returned {ret}")
    return devInfo.item()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def cuda_time(setup_fn, fn, warmup=1, repeat=3):
    """Return median elapsed milliseconds over `repeat` timed runs.
    `setup_fn` is called before each run but is not counted in elapsed time."""
    for _ in range(warmup):
        setup_fn()
        fn()
    torch.cuda.synchronize()
    result = None
    times = []
    for _ in range(repeat):
        setup_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], result


# ---------------------------------------------------------------------------
# Original pipeline benchmarks: ssfrk / spftrf / spftrs
# ---------------------------------------------------------------------------

def bench_curfp(n, k, nrhs):
    mem_gb = n * (n + 1) // 2 * 4 / 1024**3
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    [n={n}] OOM allocating RFP array ({mem_gb:.2f} GB)")
        return None, None, None, mem_gb

    t_sfrk, _ = cuda_time(lambda: None, lambda: curfp.ssfrk(A, C))

    curfp.add_to_diagonal(C, 1.0)
    t_chol, _ = cuda_time(
        lambda: (curfp.ssfrk(A, C), curfp.add_to_diagonal(C, 1.0)),
        lambda: curfp.spftrf(C, check=False),
    )
    t_solve, _ = cuda_time(
        lambda: (curfp.ssfrk(A, C), curfp.add_to_diagonal(C, 1.0), curfp.spftrf(C, check=False)),
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
        return t_torch_sgemm, t_cublas_sgemm, t_ssyrk, t_torch_chol, t_cusolver_chol, t_solve, mat_gb

    t_torch_sgemm, _ = cuda_time(lambda: None, lambda: torch.mm(A, A.t(), out=S))
    t_cublas_sgemm, _ = cuda_time(lambda: None, lambda: cublas_sgemm(n, k, 1.0, A, n, 0.0, S, n))
    t_ssyrk, _ = cuda_time(lambda: None, lambda: cublas_ssyrk(n, k, 1.0, A, n, 0.0, S, n))

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
    return t_torch_sgemm, t_cublas_sgemm, t_ssyrk, t_torch_chol, t_cusolver_chol, t_solve, mem_gb


# ---------------------------------------------------------------------------
# New benchmarks: spftri, slansf, ssfmv, strttf, stfttr, spfcon
# ---------------------------------------------------------------------------

def bench_spftri(n, k):
    """spftri (RFP inverse) vs cusolverDnSpotri and torch.linalg.inv (dense)."""
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
        S  = torch.zeros(n, n, dtype=torch.float32, device="cuda")

        def setup_chol():
            torch.mm(A2, A2.t(), out=S)
            S.diagonal().add_(1.0)
            cusolver_spotrf(n, S)

        setup_chol()
        # cusolverDnSpotri: in-place inverse from Cholesky factor
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


def bench_slansf(n, k):
    """slansf 1-norm and Frobenius vs torch.linalg.matrix_norm on full matrix."""
    t_1 = t_fro = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        t_1,   _ = cuda_time(lambda: None, lambda: curfp.slansf(C, "1"))
        t_fro, _ = cuda_time(lambda: None, lambda: curfp.slansf(C, "F"))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_d1 = t_dfro = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S  = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        t_d1,   _ = cuda_time(lambda: None, lambda: torch.linalg.matrix_norm(S, 1))
        t_dfro, _ = cuda_time(lambda: None, lambda: torch.linalg.matrix_norm(S, "fro"))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_1, t_fro, t_d1, t_dfro


def bench_ssfmv(n, k):
    """ssfmv (RFP symmetric matvec) vs cublasSsymv and torch.mv on full n×n matrix."""
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

    t_symv = t_mv = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S  = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        x2 = torch.randn(n, dtype=torch.float32, device="cuda")
        y2 = torch.empty(n, dtype=torch.float32, device="cuda")
        # cublasSsymv: reads only the lower triangle of S (same memory saving as RFP in principle)
        t_symv, _ = cuda_time(lambda: None, lambda: cublas_ssymv(n, S, x2, y2))
        t_mv,   _ = cuda_time(lambda: None, lambda: torch.mv(S, x2))
        del A2, S, x2, y2
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_symv, t_mv


def bench_strttf(n, k):
    """strttf (full tri → RFP) vs torch.triu (full tri → full n×n)."""
    try:
        A   = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S   = torch.mm(A, A.t())
        S.diagonal().add_(1.0)
        tri = torch.triu(S)
        del S
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None, None

    t_rfp = None
    try:
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.strttf(tri, uplo="U"))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    t_dense = None
    try:
        # torch.triu on n×n — keeps n×n output (2× memory of RFP output)
        t_dense, _ = cuda_time(lambda: None, lambda: torch.triu(tri))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass

    del A, tri
    torch.cuda.empty_cache()
    return t_rfp, t_dense


def bench_stfttr(n, k):
    """stfttr (RFP → full tri) vs torch.tril on a full n×n symmetric matrix."""
    # curfp.stfttr allocates an n×n output internally — guard for large n
    t_rfp = None
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

    t_dense = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S  = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        # torch.tril as a representative dense triangle operation
        t_dense, _ = cuda_time(lambda: None, lambda: torch.tril(S))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dense


def bench_spfcon(n, k):
    """spfcon (O(n²) condition estimate) vs torch.linalg.cond (O(n³) SVD)."""
    t_rfp = None
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        anorm = curfp.slansf(C, "1")
        curfp.spftrf(C, check=False)
        # spfcon is non-destructive — no setup needed between runs
        t_rfp, _ = cuda_time(lambda: None, lambda: curfp.spfcon(C, anorm))
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()

    t_dense = None
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S  = torch.mm(A2, A2.t())
        S.diagonal().add_(1.0)
        # torch.linalg.cond(p=1) ≈ norm(A,1) * norm(inv(A),1) — needs inverse
        t_dense, _ = cuda_time(lambda: None, lambda: torch.linalg.cond(S, 1))
        del A2, S
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()

    return t_rfp, t_dense


# ---------------------------------------------------------------------------
# Helpers (defined at module level so they're available when imported)
# ---------------------------------------------------------------------------

def _ms1(v, w=9):
    """Format milliseconds value; show OOM if None. Used in pipeline table."""
    return f"{v:>{w}.1f}ms" if v is not None else f"{'OOM':>{w+2}}"


def _ms2(v):
    """Format milliseconds value; show OOM if None. Used in extras table."""
    return f"{v:>8.2f}ms" if v is not None else f"{'OOM':>10}"


def _nn(results, idx):
    """Filter (n, val) pairs where val is not None."""
    return [(r[0], r[idx]) for r in results if r[idx] is not None]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 56000, 65536, 80000]
    k    = 1024   # rank of update
    nrhs = 64     # right-hand sides for spftrs

    print(
        f"Benchmark: curfp (RFP) vs dense — all ops, float32, rank-{k} update, {nrhs} rhs"
    )

    # ---- Table 1: pipeline (ssfrk / spftrf / spftrs) -------------------------
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
        t_pt_sgemm, t_cb_sgemm, t_ssyrk, t_torch_chol, t_cusolver_chol, t_solve_dns, mem_dns = bench_torch(n, k, nrhs)
        torch.cuda.empty_cache()

        pipeline_results.append((n, t_sfrk, t_chol_rfp, t_solve_rfp, mem_rfp,
                                  t_ssyrk, t_cb_sgemm, t_pt_sgemm,
                                  t_cusolver_chol, t_torch_chol, t_solve_dns, mem_dns))

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

    # ---- Table 2: remaining ops -----------------------------------------------
    print()
    print(
        f"{'n':>6}  {'spftri rfp':>11} {'potri dns':>10} {'inv dns':>9}"
        f"  {'slansf 1':>9} {'slansf F':>9} {'norm1 dns':>10} {'normF dns':>10}"
        f"  {'ssfmv rfp':>10} {'symv dns':>9} {'mv dense':>9}"
        f"  {'strttf':>8} {'triu dns':>9}"
        f"  {'stfttr':>8} {'tril dns':>9}"
        f"  {'spfcon':>8} {'cond dns':>9}"
    )
    print("-" * 205)

    extra_results = []

    for n in sizes:
        t_spftri_r, t_potri_d, t_inv_d = bench_spftri(n, k)
        torch.cuda.empty_cache()
        t_s1, t_sf, t_d1, t_df   = bench_slansf(n, k)
        torch.cuda.empty_cache()
        t_ssfmv_r, t_symv_d, t_mv_d = bench_ssfmv(n, k)
        torch.cuda.empty_cache()
        t_strttf_r, t_strttf_d   = bench_strttf(n, k)
        torch.cuda.empty_cache()
        t_stfttr_r, t_stfttr_d   = bench_stfttr(n, k)
        torch.cuda.empty_cache()
        t_spfcon_r, t_spfcon_d   = bench_spfcon(n, k)
        torch.cuda.empty_cache()

        # Index map:
        #  0:n  1:spftri_r  2:potri_d  3:inv_d
        #  4:s1  5:sf  6:d1  7:df
        #  8:ssfmv_r  9:symv_d  10:mv_d
        #  11:strttf_r  12:strttf_d
        #  13:stfttr_r  14:stfttr_d
        #  15:spfcon_r  16:spfcon_d
        extra_results.append((n,
            t_spftri_r, t_potri_d, t_inv_d,
            t_s1, t_sf, t_d1, t_df,
            t_ssfmv_r, t_symv_d, t_mv_d,
            t_strttf_r, t_strttf_d,
            t_stfttr_r, t_stfttr_d,
            t_spfcon_r, t_spfcon_d))

        print(
            f"{n:>6}"
            f"  {_ms2(t_spftri_r)} {_ms2(t_potri_d)} {_ms2(t_inv_d)}"
            f"  {_ms2(t_s1)} {_ms2(t_sf)} {_ms2(t_d1)} {_ms2(t_df)}"
            f"  {_ms2(t_ssfmv_r)} {_ms2(t_symv_d)} {_ms2(t_mv_d)}"
            f"  {_ms2(t_strttf_r)} {_ms2(t_strttf_d)}"
            f"  {_ms2(t_stfttr_r)} {_ms2(t_stfttr_d)}"
            f"  {_ms2(t_spfcon_r)} {_ms2(t_spfcon_d)}"
        )

    # ---------------------------------------------------------------------------
    # Plot: 2 rows × 5 columns
    # ---------------------------------------------------------------------------

    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
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

    # ---- Row 0, Col 0: ssfrk ---------------------------------------------------
    ax = axes[0, 0]
    pairs = _nn(pipeline_results, 1)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp ssfrk (RFP)")
    pairs = _nn(pipeline_results, 5)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="cublas ssyrk")
    pairs = _nn(pipeline_results, 6)
    if pairs: ax.plot(*zip(*pairs), "^:", color="darkorange", label="cublas sgemm")
    pairs = _nn(pipeline_results, 7)
    if pairs: ax.plot(*zip(*pairs), "d-.", color="green", label="torch mm")
    style_ax(ax, "ssfrk — rank-k update")

    # ---- Row 0, Col 1: spftrf ---------------------------------------------------
    ax = axes[0, 1]
    pairs = _nn(pipeline_results, 2)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp spftrf (RFP)")
    pairs = _nn(pipeline_results, 8)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="cusolver spotrf")
    pairs = _nn(pipeline_results, 9)
    if pairs: ax.plot(*zip(*pairs), "^:", color="darkorange", label="torch cholesky")
    style_ax(ax, "spftrf — Cholesky factorization")

    # ---- Row 0, Col 2: spftrs ---------------------------------------------------
    ax = axes[0, 2]
    pairs = _nn(pipeline_results, 3)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label=f"curfp spftrs (RFP, {nrhs} rhs)")
    pairs = _nn(pipeline_results, 10)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label=f"torch.cholesky_solve ({nrhs} rhs)")
    style_ax(ax, "spftrs — triangular solve")

    # ---- Row 0, Col 3: spftri ---------------------------------------------------
    # Indices: 1=rfp, 2=potri(cusolver), 3=inv(torch)
    ax = axes[0, 3]
    pairs = _nn(extra_results, 1)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp spftri (RFP)")
    pairs = _nn(extra_results, 2)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="cusolver potri (dense Cholesky)")
    pairs = _nn(extra_results, 3)
    if pairs: ax.plot(*zip(*pairs), "^:", color="darkorange", label="torch.linalg.inv (dense LU)")
    style_ax(ax, "spftri — matrix inversion")

    # ---- Row 0, Col 4: memory ---------------------------------------------------
    ax = axes[0, 4]
    ax.plot(ns_all, [r[4] for r in pipeline_results], "o-", color="steelblue", label="RFP n(n+1)/2 floats")
    pairs = [(r[0], r[11]) for r in pipeline_results if r[11] is not None]
    if pairs:
        ns_d, mem_d = zip(*pairs)
        ax.plot(ns_d, mem_d, "s--", color="tomato", label="Dense n×n floats")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Matrix memory footprint")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns_all)
    ax.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # ---- Row 1, Col 0: slansf ---------------------------------------------------
    # Indices: 4=s1, 5=sf, 6=d1, 7=df
    ax = axes[1, 0]
    pairs = _nn(extra_results, 4)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp slansf 1-norm (RFP)")
    pairs = _nn(extra_results, 5)
    if pairs: ax.plot(*zip(*pairs), "o--", color="cornflowerblue", label="curfp slansf Frobenius (RFP)")
    pairs = _nn(extra_results, 6)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="torch matrix_norm 1 (dense)")
    pairs = _nn(extra_results, 7)
    if pairs: ax.plot(*zip(*pairs), "^:", color="darkorange", label="torch matrix_norm fro (dense)")
    style_ax(ax, "slansf — matrix norm")

    # ---- Row 1, Col 1: ssfmv ---------------------------------------------------
    # Indices: 8=rfp, 9=symv(cublas), 10=mv(torch)
    ax = axes[1, 1]
    pairs = _nn(extra_results, 8)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp ssfmv (RFP)")
    pairs = _nn(extra_results, 9)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="cublas ssymv (dense lower tri)")
    pairs = _nn(extra_results, 10)
    if pairs: ax.plot(*zip(*pairs), "^:", color="darkorange", label="torch.mv (dense full)")
    style_ax(ax, "ssfmv — symmetric matvec")

    # ---- Row 1, Col 2: strttf ---------------------------------------------------
    # Indices: 11=rfp, 12=triu(torch)
    ax = axes[1, 2]
    pairs = _nn(extra_results, 11)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp strttf (→ n*(n+1)/2 RFP)")
    pairs = _nn(extra_results, 12)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="torch.triu (→ n×n dense)")
    style_ax(ax, "strttf — full tri → RFP")

    # ---- Row 1, Col 3: stfttr ---------------------------------------------------
    # Indices: 13=rfp, 14=tril(torch)
    ax = axes[1, 3]
    pairs = _nn(extra_results, 13)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp stfttr (RFP → n×n tri)")
    pairs = _nn(extra_results, 14)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="torch.tril (dense n×n → n×n)")
    style_ax(ax, "stfttr — RFP → full tri")

    # ---- Row 1, Col 4: spfcon ---------------------------------------------------
    # Indices: 15=rfp, 16=cond(torch)
    ax = axes[1, 4]
    pairs = _nn(extra_results, 15)
    if pairs: ax.plot(*zip(*pairs), "o-", color="steelblue", label="curfp spfcon (O(n²) estimate)")
    pairs = _nn(extra_results, 16)
    if pairs: ax.plot(*zip(*pairs), "s--", color="tomato", label="torch.linalg.cond (O(n³) SVD)")
    style_ax(ax, "spfcon — condition number")

    plt.tight_layout()
    out = "benchmark.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
