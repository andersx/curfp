"""
Scaled-down benchmark: curfp float32 (RFP) vs PyTorch / cuBLAS / cuSOLVER.

Three operations only:
  ssfrk  — rank-k update into RFP        vs cublasSsyrk / cublasSgemm / torch.mm
  spftrf — Cholesky factorization         vs cusolverSpotrf / torch.linalg.cholesky
  spftrs — Cholesky triangular solve      vs torch.cholesky_solve

Memory discipline:
  - Every bench_* function allocates all tensors locally and explicitly deletes
    them before returning (including cuSOLVER workspace).
  - torch.cuda.empty_cache() + gc.collect() are called after every bench call.
  - For n > LARGE_N_THRESHOLD only one timed trial is run (warmup still runs once).

Colour convention:
  Blue  (#2166ac)  — curfp RFP
  Green (#238b45 / #005a32)  — cuBLAS / cuSOLVER
  Red   (#cb181d)  — PyTorch / torch
"""

import gc
import csv
import ctypes
import atexit

import torch
import curfp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Large-n threshold: only 1 timed trial above this
# ---------------------------------------------------------------------------
LARGE_N_THRESHOLD = 65536

# ---------------------------------------------------------------------------
# Load libraries
# ---------------------------------------------------------------------------
_libcublas = ctypes.CDLL("libcublas.so.12", use_errno=True)
_libcusolver = ctypes.CDLL("libcusolver.so.11", use_errno=True)

# ---------------------------------------------------------------------------
# cuBLAS handle  (created once, destroyed at exit)
# ---------------------------------------------------------------------------
_libcublas.cublasCreate_v2.restype = ctypes.c_int
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcublas.cublasDestroy_v2.restype = ctypes.c_int
_libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

_cublas_handle = ctypes.c_void_p()
assert _libcublas.cublasCreate_v2(ctypes.byref(_cublas_handle)) == 0

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


@atexit.register
def _destroy_cublas():
    if _cublas_handle:
        _libcublas.cublasDestroy_v2(_cublas_handle)


# ---------------------------------------------------------------------------
# cuSOLVER handle  (created once, destroyed at exit)
# ---------------------------------------------------------------------------
_libcusolver.cusolverDnCreate.restype = ctypes.c_int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcusolver.cusolverDnDestroy.restype = ctypes.c_int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]

CUSOLVER_FILL_MODE_LOWER = 0
_cusolver_handle = ctypes.c_void_p()
assert _libcusolver.cusolverDnCreate(ctypes.byref(_cusolver_handle)) == 0


@atexit.register
def _destroy_cusolver():
    if _cusolver_handle:
        _libcusolver.cusolverDnDestroy(_cusolver_handle)


# ---------------------------------------------------------------------------
# cublasSgemm  (C = alpha * A * A^T + beta * C,  OP_N × OP_T)
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
# cublasSsyrk  (C = alpha * A * A^T + beta * C,  lower triangle)
# A is (n, k) row-major = k×n col-major with lda=n; OP_N with lda=n
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
# cusolverDnSpotrf  (Cholesky factorisation, dense lower triangle)
# cusolverDnSpotrs  (Cholesky triangular solve, dense lower triangle)
# Workspace is allocated locally per call — no global cache.
# ---------------------------------------------------------------------------
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


def cusolver_spotrf_with_workspace(n, S, workspace, devInfo):
    """Run spotrf using caller-supplied workspace and devInfo tensors."""
    lwork = workspace.numel()
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


def spotrf_query_lwork(n, S):
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
        raise RuntimeError(f"cusolverDnSpotrf_bufferSize returned {ret}")
    return lwork.value


# cusolverDnSpotrs  — no workspace, signature is:
# (handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) -> status
_libcusolver.cusolverDnSpotrs.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrs.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,  # uplo
    ctypes.c_int,  # n
    ctypes.c_int,  # nrhs
    ctypes.c_void_p,  # A (Cholesky factor, lower triangle)
    ctypes.c_int,  # lda
    ctypes.c_void_p,  # B (in: rhs, out: solution)
    ctypes.c_int,  # ldb
    ctypes.c_void_p,  # devInfo
]


def cusolver_spotrs(n, nrhs, A, B, devInfo):
    """Cholesky triangular solve. A must already contain the Cholesky factor
    (lower triangle). B is overwritten with the solution in-place."""
    ret = _libcusolver.cusolverDnSpotrs(
        _cusolver_handle,
        CUSOLVER_FILL_MODE_LOWER,
        n,
        nrhs,
        ctypes.c_void_p(A.data_ptr()),
        n,
        ctypes.c_void_p(B.data_ptr()),
        n,
        ctypes.c_void_p(devInfo.data_ptr()),
    )
    if ret != 0:
        raise RuntimeError(f"cusolverDnSpotrs returned {ret}")
    return devInfo.item()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def cuda_time(setup_fn, fn, warmup=1, repeat=3):
    """Return median elapsed milliseconds.  repeat=1 skips median logic."""
    for _ in range(warmup):
        setup_fn()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        setup_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        times.append(t)
        # Explicitly release CUDA events
        del start, end
    times.sort()
    return times[len(times) // 2]


def _n_repeats(n):
    """1 trial for large n, 3 trials otherwise."""
    return 1 if n > LARGE_N_THRESHOLD else 3


# ---------------------------------------------------------------------------
# bench_sfrk  — rank-k update
#   curfp ssfrk  vs  cublasSsyrk  vs  cublasSgemm  vs  torch.mm
# ---------------------------------------------------------------------------
def bench_sfrk(n, k):
    repeat = _n_repeats(n)
    t_rfp = t_ssyrk = t_sgemm = t_mm = None

    # --- curfp ssfrk ---
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        t_rfp = cuda_time(lambda: None, lambda: curfp.ssfrk(A, C), repeat=repeat)
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    # --- cublasSsyrk, cublasSgemm, torch.mm ---
    try:
        # A_dns is (n, k) contiguous row-major; treated as k×n col-major by cuBLAS
        A_dns = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.empty(n, n, dtype=torch.float32, device="cuda")

        t_ssyrk = cuda_time(
            lambda: None,
            lambda: cublas_ssyrk(n, k, 1.0, A_dns, n, 0.0, S, n),
            repeat=repeat,
        )
        t_sgemm = cuda_time(
            lambda: None,
            lambda: cublas_sgemm(n, k, 1.0, A_dns, n, 0.0, S, n),
            repeat=repeat,
        )
        t_mm = cuda_time(
            lambda: None,
            lambda: torch.mm(A_dns, A_dns.t(), out=S),
            repeat=repeat,
        )
        del A_dns, S
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    return t_rfp, t_ssyrk, t_sgemm, t_mm


# ---------------------------------------------------------------------------
# bench_chol  — Cholesky factorisation
#   curfp spftrf  vs  cusolverSpotrf  vs  torch.linalg.cholesky
# ---------------------------------------------------------------------------
def bench_chol(n, k):
    repeat = _n_repeats(n)
    t_rfp = t_cusolver = t_torch = None

    # --- curfp spftrf ---
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")

        def setup_rfp():
            curfp.ssfrk(A, C)
            curfp.add_to_diagonal(C, 1.0)

        t_rfp = cuda_time(
            setup_rfp, lambda: curfp.spftrf(C, check=False), repeat=repeat
        )
        del A, C
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    # --- cusolverSpotrf ---
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.empty(n, n, dtype=torch.float32, device="cuda")

        # Query lwork using S directly (bufferSize only needs a valid pointer,
        # not a PD matrix), so no extra n×n temporary is needed.
        lwork = spotrf_query_lwork(n, S)
        torch.cuda.empty_cache()

        workspace = torch.empty(lwork, dtype=torch.float32, device="cuda")
        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")

        def setup_cusolver():
            torch.mm(A2, A2.t(), out=S)
            S.diagonal().add_(1.0)

        t_cusolver = cuda_time(
            setup_cusolver,
            lambda: cusolver_spotrf_with_workspace(n, S, workspace, devInfo),
            repeat=repeat,
        )
        del A2, S, workspace, devInfo
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    # --- torch.linalg.cholesky ---
    try:
        A3 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S2 = torch.empty(n, n, dtype=torch.float32, device="cuda")

        def setup_torch():
            torch.mm(A3, A3.t(), out=S2)
            S2.diagonal().add_(1.0)

        t_torch = cuda_time(
            setup_torch,
            lambda: torch.linalg.cholesky(S2, out=S2),
            repeat=repeat,
        )
        del A3, S2
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    return t_rfp, t_cusolver, t_torch


# ---------------------------------------------------------------------------
# bench_solve  — Cholesky triangular solve
#   curfp spftrs  vs  cusolverSpotrs  vs  torch.cholesky_solve
# ---------------------------------------------------------------------------
def bench_solve(n, k, nrhs):
    repeat = _n_repeats(n)
    t_rfp = t_cusolver = t_torch = None

    # --- curfp spftrs ---
    try:
        A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")

        def setup_rfp():
            curfp.ssfrk(A, C)
            curfp.add_to_diagonal(C, 1.0)
            curfp.spftrf(C, check=False)

        t_rfp = cuda_time(setup_rfp, lambda: curfp.spftrs(C, B), repeat=repeat)
        del A, C, B
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    # --- cusolverSpotrs ---
    # spotrs takes no workspace. Build the Cholesky factor into S once;
    # each timed call restores B2 from B_ref and solves in-place.
    try:
        A2 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S = torch.empty(n, n, dtype=torch.float32, device="cuda")
        B2 = torch.empty(n, nrhs, dtype=torch.float32, device="cuda")
        B_ref = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")

        # Build Cholesky factor once into S (stays fixed across timed calls).
        # Name and delete the potrf workspace before proceeding.
        torch.mm(A2, A2.t(), out=S)
        S.diagonal().add_(1.0)
        pre_lwork = spotrf_query_lwork(n, S)
        torch.cuda.empty_cache()
        pre_ws = torch.empty(pre_lwork, dtype=torch.float32, device="cuda")
        pre_info = torch.zeros(1, dtype=torch.int32, device="cuda")
        cusolver_spotrf_with_workspace(n, S, pre_ws, pre_info)
        del pre_ws, pre_info
        torch.cuda.empty_cache()

        devInfo = torch.zeros(1, dtype=torch.int32, device="cuda")

        # setup: restore B2 from B_ref so each trial starts with fresh rhs
        def setup_cusolver():
            B2.copy_(B_ref)

        t_cusolver = cuda_time(
            setup_cusolver,
            lambda: cusolver_spotrs(n, nrhs, S, B2, devInfo),
            repeat=repeat,
        )
        del A2, S, B2, B_ref, devInfo
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    # --- torch.cholesky_solve ---
    try:
        A3 = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
        S2 = torch.empty(n, n, dtype=torch.float32, device="cuda")
        B3 = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")

        def setup_torch():
            torch.mm(A3, A3.t(), out=S2)
            S2.diagonal().add_(1.0)
            torch.linalg.cholesky(S2, out=S2)

        t_torch = cuda_time(
            setup_torch,
            lambda: torch.cholesky_solve(B3, S2),
            repeat=repeat,
        )
        del A3, S2, B3
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    torch.cuda.empty_cache()
    gc.collect()

    return t_rfp, t_cusolver, t_torch


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _ms(v, w=10):
    return f"{v:>{w}.2f}ms" if v is not None else f"{'OOM':>{w + 2}}"


def _nn(results, idx):
    return [(r[0], r[idx]) for r in results if r[idx] is not None]


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_RFP_BLUE = "#2166ac"
C_CB_GREEN = "#238b45"
C_CS_GREEN = "#005a32"
C_PT_RED = "#cb181d"
C_CB_GREEN2 = "#74c476"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144]
    sizes = [1024, 2048, 4096, 8192, 16384, 32768,56000, 65536, 80000]
    k = 1024
    nrhs = 64

    print(
        f"Benchmark (mini, float32): curfp RFP vs cuBLAS / cuSOLVER / torch"
        f"  —  rank-{k} update, {nrhs} rhs"
    )
    print(f"  n > {LARGE_N_THRESHOLD}: 1 timed trial (warmup still runs)")
    print()

    hdr = (
        f"{'n':>8}"
        f"  {'ssfrk rfp':>10} {'ssyrk cb':>10} {'sgemm cb':>10} {'mm torch':>10}"
        f"  {'spftrf rfp':>11} {'spotrf cs':>11} {'chol torch':>11}"
        f"  {'spftrs rfp':>11} {'spotrs cs':>11} {'solve torch':>12}"
        f"  {'rfp mem GB':>11} {'dns mem GB':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []

    for n in sizes:
        t_sfrk, t_ssyrk, t_sgemm, t_mm = bench_sfrk(n, k)
        torch.cuda.empty_cache()
        gc.collect()

        t_chol_rfp, t_cusolver, t_torch_chol = bench_chol(n, k)
        torch.cuda.empty_cache()
        gc.collect()

        t_solve_rfp, t_cusolver_solve, t_torch_solve = bench_solve(n, k, nrhs)
        torch.cuda.empty_cache()
        gc.collect()

        rfp_mem_gb = n * (n + 1) // 2 * 4 / 1024**3
        dns_mem_gb = n * n * 4 / 1024**3

        results.append(
            (
                n,
                t_sfrk,
                t_ssyrk,
                t_sgemm,
                t_mm,
                t_chol_rfp,
                t_cusolver,
                t_torch_chol,
                t_solve_rfp,
                t_cusolver_solve,
                t_torch_solve,
                rfp_mem_gb,
                dns_mem_gb,
            )
        )

        trial_tag = " *" if n > LARGE_N_THRESHOLD else "  "
        print(
            f"{n:>8}{trial_tag}"
            f"  {_ms(t_sfrk)} {_ms(t_ssyrk)} {_ms(t_sgemm)} {_ms(t_mm)}"
            f"  {_ms(t_chol_rfp)} {_ms(t_cusolver)} {_ms(t_torch_chol)}"
            f"  {_ms(t_solve_rfp)} {_ms(t_cusolver_solve)} {_ms(t_torch_solve)}"
            f"  {rfp_mem_gb:>9.3f} GB {dns_mem_gb:>9.3f} GB"
        )

    print()
    print("  * = 1 timed trial only (n > LARGE_N_THRESHOLD)")

    # -------------------------------------------------------------------------
    # Plot: 1 row × 3 columns
    # -------------------------------------------------------------------------
    ns_all = [r[0] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"curfp (RFP) vs dense — float32, rank-{k} update, {nrhs} rhs  "
        f"[* = 1 trial for n > {LARGE_N_THRESHOLD}]",
        fontsize=13,
    )

    def style_ax(ax, title):
        ax.set_xlabel("Matrix size n")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns_all)
        ax.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    # [0] ssfrk
    ax = axes[0]
    p = _nn(results, 1)
    if p:
        ax.plot(*zip(*p), "o-", color=C_RFP_BLUE, label="curfp ssfrk (RFP)")
    p = _nn(results, 2)
    if p:
        ax.plot(*zip(*p), "s--", color=C_CB_GREEN, label="cublas ssyrk (dense lower)")
    p = _nn(results, 3)
    if p:
        ax.plot(*zip(*p), "^--", color=C_CB_GREEN2, label="cublas sgemm (dense full)")
    p = _nn(results, 4)
    if p:
        ax.plot(*zip(*p), "v:", color=C_PT_RED, label="torch.mm (dense full)")
    style_ax(ax, "ssfrk — rank-k update")

    # [1] spftrf
    ax = axes[1]
    p = _nn(results, 5)
    if p:
        ax.plot(*zip(*p), "o-", color=C_RFP_BLUE, label="curfp spftrf (RFP)")
    p = _nn(results, 6)
    if p:
        ax.plot(*zip(*p), "D-.", color=C_CS_GREEN, label="cusolver spotrf (dense)")
    p = _nn(results, 7)
    if p:
        ax.plot(*zip(*p), "v:", color=C_PT_RED, label="torch.linalg.cholesky (dense)")
    style_ax(ax, "spftrf — Cholesky factorisation")

    # [2] spftrs
    ax = axes[2]
    p = _nn(results, 8)
    if p:
        ax.plot(
            *zip(*p), "o-", color=C_RFP_BLUE, label=f"curfp spftrs (RFP, {nrhs} rhs)"
        )
    p = _nn(results, 9)
    if p:
        ax.plot(
            *zip(*p),
            "D-.",
            color=C_CS_GREEN,
            label=f"cusolver spotrs (dense, {nrhs} rhs)",
        )
    p = _nn(results, 10)
    if p:
        ax.plot(
            *zip(*p), "v:", color=C_PT_RED, label=f"torch.cholesky_solve ({nrhs} rhs)"
        )
    style_ax(ax, "spftrs — Cholesky solve")

    plt.tight_layout()
    out_png = "benchmark_mini32.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_png}")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------
    out_csv = "benchmark_mini32.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "ssfrk_rfp_ms",
                "ssyrk_cb_ms",
                "sgemm_cb_ms",
                "mm_torch_ms",
                "spftrf_rfp_ms",
                "spotrf_cusolver_ms",
                "cholesky_torch_ms",
                "spftrs_rfp_ms",
                "spotrs_cusolver_ms",
                "solve_torch_ms",
                "rfp_mem_gb",
                "dns_mem_gb",
            ]
        )
        for r in results:
            w.writerow(r)
    print(f"Timings saved to {out_csv}")
