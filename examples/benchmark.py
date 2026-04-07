"""
Benchmark: curfp (RFP format) vs PyTorch (dense) for the full solver pipeline:
  rank-k update → Cholesky → triangular solve.

curfp path:
  ssfrk  — rank-k update into RFP-packed storage   (cuBLAS ssyrk/sgemm internally)
  spftrf — Cholesky factorization of RFP matrix     (cuSOLVER spotrf internally)
  spftrs — triangular solve using RFP Cholesky factor

PyTorch / cuBLAS path:
  cublas ssyrk / sgemm — dense rank-k update
  cusolver spotrf      — dense Cholesky (cuSOLVER SPOTRF on full n×n matrix)
  torch.cholesky_solve — dense triangular solve
"""

import ctypes
import ctypes.util

import torch
import curfp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Minimal ctypes wrapper around cublasSsyrk_v2
# ---------------------------------------------------------------------------
_libcublas = ctypes.CDLL("libcublas.so.12", use_errno=True)

# cublasCreate / cublasDestroy
_libcublas.cublasCreate_v2.restype = ctypes.c_int
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcublas.cublasDestroy_v2.restype = ctypes.c_int
_libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

# cublasSsyrk_v2
_libcublas.cublasSsyrk_v2.restype = ctypes.c_int
_libcublas.cublasSsyrk_v2.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,  # uplo  (CUBLAS_FILL_MODE_LOWER = 1)
    ctypes.c_int,  # trans (CUBLAS_OP_T = 1)
    ctypes.c_int,  # n
    ctypes.c_int,  # k
    ctypes.POINTER(ctypes.c_float),  # alpha (host ptr)
    ctypes.c_void_p,  # A     (device ptr)
    ctypes.c_int,  # lda
    ctypes.POINTER(ctypes.c_float),  # beta  (host ptr)
    ctypes.c_void_p,  # C     (device ptr)
    ctypes.c_int,  # ldc
]

_cublas_handle = ctypes.c_void_p()
assert _libcublas.cublasCreate_v2(ctypes.byref(_cublas_handle)) == 0

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

# cublasSgemm_v2
_libcublas.cublasSgemm_v2.restype = ctypes.c_int
_libcublas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,  # transa
    ctypes.c_int,  # transb
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.c_int,  # k
    ctypes.POINTER(ctypes.c_float),  # alpha (host ptr)
    ctypes.c_void_p,  # A (device ptr)
    ctypes.c_int,  # lda
    ctypes.c_void_p,  # B (device ptr)
    ctypes.c_int,  # ldb
    ctypes.POINTER(ctypes.c_float),  # beta (host ptr)
    ctypes.c_void_p,  # C (device ptr)
    ctypes.c_int,  # ldc
]


def cublas_sgemm(
    n: int,
    k: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    beta: float,
    C: torch.Tensor,
    ldc: int,
) -> None:
    """C = alpha * A * A^T + beta * C  via cublasSgemm_v2.
    A is (n,k) col-major with lda=n. C = A * A^T = (n,k) * (k,n) = (n,n).
    cuBLAS: transa=N, transb=T, m=n, n=n, k=k."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSgemm_v2(
        _cublas_handle,
        CUBLAS_OP_N,  # transa: A as-is (n,k)
        CUBLAS_OP_T,  # transb: A^T → (k,n)^T = (n,k)^T... gives (k,n)
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


def cublas_ssyrk(
    n: int,
    k: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    beta: float,
    C: torch.Tensor,
    ldc: int,
) -> None:
    """Call cublasSsyrk_v2 directly. A is (n,k) col-major with lda=n.
    trans=N: C = alpha * A * A^T + beta * C."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    ret = _libcublas.cublasSsyrk_v2(
        _cublas_handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,  # A * A^T with A as (n,k) col-major
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
# Minimal ctypes wrapper around cusolverDnSpotrf
# ---------------------------------------------------------------------------
_libcusolver = ctypes.CDLL("libcusolver.so.11", use_errno=True)

_libcusolver.cusolverDnCreate.restype = ctypes.c_int
_libcusolver.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcusolver.cusolverDnDestroy.restype = ctypes.c_int
_libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverDnSpotrf_bufferSize.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrf_bufferSize.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,  # uplo
    ctypes.c_int,  # n
    ctypes.c_void_p,  # A (device ptr)
    ctypes.c_int,  # lda
    ctypes.POINTER(ctypes.c_int),  # lwork (output)
]

_libcusolver.cusolverDnSpotrf.restype = ctypes.c_int
_libcusolver.cusolverDnSpotrf.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,  # uplo
    ctypes.c_int,  # n
    ctypes.c_void_p,  # A (device ptr)
    ctypes.c_int,  # lda
    ctypes.c_void_p,  # workspace (device ptr)
    ctypes.c_int,  # lwork
    ctypes.c_void_p,  # devInfo (device ptr)
]

CUSOLVER_FILL_MODE_LOWER = 0

_cusolver_handle = ctypes.c_void_p()
assert _libcusolver.cusolverDnCreate(ctypes.byref(_cusolver_handle)) == 0

# Cache workspace tensors to avoid repeated allocation
_cusolver_workspace = {}


def cusolver_spotrf(n: int, S: torch.Tensor) -> int:
    """Call cusolverDnSpotrf directly. S is n×n column-major (lower triangle), modified in-place."""
    # Get or allocate workspace
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
        assert ret == 0, f"cusolverDnSpotrf_bufferSize returned {ret}"
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
    assert ret == 0, f"cusolverDnSpotrf returned {ret}"
    return devInfo.item()


# ---------------------------------------------------------------------------


def cuda_time(setup_fn, fn, warmup=1, repeat=3):
    """Return median elapsed milliseconds over `repeat` timed runs.

    `setup_fn` is called before each run (warmup and timed) but is not counted
    in the elapsed time. Use it to restore in-place buffers between runs.
    """
    result = None
    for _ in range(warmup):
        setup_fn()
        result = fn()
    torch.cuda.synchronize()

    times = []
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


def bench_curfp(n, k, nrhs):
    A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
    C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
    B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")

    def do_ssfrk():
        curfp.ssfrk(A, C)

    def ssfrk_then_diag():
        do_ssfrk()
        curfp.add_to_diagonal(C, 1.0)

    def chol_then_restore():
        curfp.ssfrk(A, C)
        curfp.add_to_diagonal(C, 1.0)
        curfp.spftrf(C, check=False)

    t_sfrk, _ = cuda_time(lambda: None, do_ssfrk)

    curfp.add_to_diagonal(C, 1.0)
    t_chol, _ = cuda_time(
        ssfrk_then_diag,
        lambda: curfp.spftrf(C, check=False),
    )

    t_solve, _ = cuda_time(
        chol_then_restore,
        lambda: curfp.spftrs(C, B),
    )

    mem_gb = C.numel() * C.element_size() / 1024**3
    del A, C, B
    return t_sfrk, t_chol, t_solve, mem_gb


def bench_torch(n, k, nrhs):
    """Returns (t_torch_sgemm, t_cublas_sgemm, t_ssyrk, t_torch_chol, t_cusolver_chol, t_solve, mem_gb).
    Any field that OOMs is set to None."""
    mat_gb = n * n * 4 / 1024**3
    t_torch_sgemm = t_cublas_sgemm = t_ssyrk = t_torch_chol = t_cusolver_chol = (
        t_solve
    ) = None

    try:
        # Column-major A for trans=N: (n,k) col-major with lda=n
        A = (torch.randn(k, n, dtype=torch.float32, device="cuda") / k**0.5).t()
        S = torch.empty(n, n, dtype=torch.float32, device="cuda")
        B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    [n={n}] OOM allocating dense {n}x{n} matrix ({mat_gb:.2f} GB)")
        return (
            t_torch_sgemm,
            t_cublas_sgemm,
            t_ssyrk,
            t_torch_chol,
            t_cusolver_chol,
            t_solve,
            mat_gb,
        )

    def torch_sgemm():
        # torch.mm handles non-contiguous (col-major) A correctly
        torch.mm(A, A.t(), out=S)

    def cb_sgemm():
        cublas_sgemm(n, k, 1.0, A, n, 0.0, S, n)

    def ssyrk():
        cublas_ssyrk(n, k, 1.0, A, n, 0.0, S, n)

    def ssyrk_then_diag():
        ssyrk()
        S.diagonal().add_(1.0)

    def sgemm_then_diag():
        torch_sgemm()
        S.diagonal().add_(1.0)

    t_torch_sgemm, _ = cuda_time(lambda: None, torch_sgemm)
    t_cublas_sgemm, _ = cuda_time(lambda: None, cb_sgemm)
    t_ssyrk, _ = cuda_time(lambda: None, ssyrk)

    try:
        t_cusolver_chol, info = cuda_time(
            ssyrk_then_diag,
            lambda: cusolver_spotrf(n, S),
        )
        assert info == 0, f"cusolverDnSpotrf failed at minor {info}"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    [n={n}] OOM in cusolver spotrf (matrix={mat_gb:.2f} GB)")
        t_cusolver_chol = None

    try:
        # sgemm fills both triangles — torch.linalg.cholesky may check symmetry
        t_torch_chol, _ = cuda_time(
            sgemm_then_diag,
            lambda: torch.linalg.cholesky(S, out=S),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    [n={n}] OOM in torch.linalg.cholesky (matrix={mat_gb:.2f} GB)")
        t_torch_chol = None

    # Dense triangular solve using the torch Cholesky factor
    if t_torch_chol is not None:
        # cholesky_solve expects upper triangular factor; torch.linalg.cholesky gives lower
        def restore_and_chol():
            torch_sgemm()
            S.diagonal().add_(1.0)
            torch.linalg.cholesky(S, out=S)

        try:
            t_solve, _ = cuda_time(
                restore_and_chol,
                lambda: torch.cholesky_solve(B, S),
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_solve = None

    mem_gb = S.numel() * S.element_size() / 1024**3
    del A, S, B
    return (
        t_torch_sgemm,
        t_cublas_sgemm,
        t_ssyrk,
        t_torch_chol,
        t_cusolver_chol,
        t_solve,
        mem_gb,
    )


sizes = [1024, 2048, 4096, 8192, 16384, 32768, 56000, 65536, 80000]
k = 1024  # rank of update (tall and skinny A)
nrhs = 64  # right-hand sides for the solve

print(
    f"Benchmark: curfp (RFP) vs dense (cuBLAS/cuSOLVER/PyTorch),  rank-{k} update + Cholesky + solve ({nrhs} rhs)"
)
print(
    f"{'n':>6}  {'curfp ssfrk':>12} {'curfp spftrf':>13} {'curfp spftrs':>13} {'curfp mem':>10}"
    f"  {'ssyrk':>8} {'cb sgemm':>9} {'pt sgemm':>9} {'cusol chol':>11} {'torch chol':>11} {'torch solve':>12} {'dense mem':>10}"
)
print("-" * 167)

results = []

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

    def fmt_ms(v):
        return f"{v:>9.1f}ms" if v is not None else f"{'OOM':>11}"

    torch_str = (
        f"  {fmt_ms(t_ssyrk)}"
        f"  {fmt_ms(t_cb_sgemm)}"
        f"  {fmt_ms(t_pt_sgemm)}"
        f"  {fmt_ms(t_cusolver_chol)}"
        f"  {fmt_ms(t_torch_chol)}"
        f"  {fmt_ms(t_solve_dns)}"
        f"  {mem_dns:>8.3f} GB"
    )
    results.append(
        (
            n,
            t_sfrk,  # 1
            t_chol_rfp,  # 2
            t_solve_rfp,  # 3
            mem_rfp,  # 4
            t_ssyrk,  # 5
            t_cb_sgemm,  # 6
            t_pt_sgemm,  # 7
            t_cusolver_chol,  # 8
            t_torch_chol,  # 9
            t_solve_dns,  # 10
            mem_dns,  # 11
        )
    )

    print(
        f"{n:>6}"
        f"  {t_sfrk:>10.1f}ms"
        f"  {t_chol_rfp:>11.1f}ms"
        f"  {t_solve_rfp:>11.1f}ms"
        f"  {mem_rfp:>8.3f} GB" + torch_str
    )

# --- Plot ---
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
ax1, ax2, ax3, ax4 = axes
fig.suptitle(
    f"curfp (RFP) vs dense — rank-{k} update + Cholesky + solve ({nrhs} rhs), float32",
    fontsize=13,
)

ns_all = [r[0] for r in results]
t_ssfrk = [r[1] for r in results]
t_spftrf = [r[2] for r in results]
t_spftrs = [r[3] for r in results]
mem_rfp = [r[4] for r in results]

# Each dense metric may independently be None (OOM), pair with its own n list
ns_ssyrk = [r[0] for r in results if r[5] is not None]
t_ssyrk = [r[5] for r in results if r[5] is not None]
ns_cb_sgemm = [r[0] for r in results if r[6] is not None]
t_cb_sgemm = [r[6] for r in results if r[6] is not None]
ns_pt_sgemm = [r[0] for r in results if r[7] is not None]
t_pt_sgemm = [r[7] for r in results if r[7] is not None]
ns_cusolver = [r[0] for r in results if r[8] is not None]
t_cusolver = [r[8] for r in results if r[8] is not None]
ns_torch_chol = [r[0] for r in results if r[9] is not None]
t_torch_chol = [r[9] for r in results if r[9] is not None]
ns_torch_solv = [r[0] for r in results if r[10] is not None]
t_torch_solv = [r[10] for r in results if r[10] is not None]
ns_mem_dns = [r[0] for r in results if r[5] is not None]
mem_dns = [r[11] for r in results if r[5] is not None]


def style_ax(ax, title, ylabel):
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns_all)
    ax.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


# Panel 1: rank-k update
ax1.plot(ns_all, t_ssfrk, "o-", color="steelblue", label="curfp ssfrk (RFP)")
ax1.plot(ns_ssyrk, t_ssyrk, "s--", color="tomato", label="cublas ssyrk (dense)")
ax1.plot(
    ns_cb_sgemm, t_cb_sgemm, "^:", color="darkorange", label="cublas sgemm (dense)"
)
ax1.plot(ns_pt_sgemm, t_pt_sgemm, "d-.", color="green", label="torch sgemm (dense)")
style_ax(ax1, "Rank-k update", "Time (ms)")

# Panel 2: Cholesky
ax2.plot(ns_all, t_spftrf, "o-", color="steelblue", label="curfp spftrf (RFP)")
ax2.plot(
    ns_cusolver, t_cusolver, "s--", color="tomato", label="cusolver spotrf (dense)"
)
ax2.plot(
    ns_torch_chol,
    t_torch_chol,
    "^:",
    color="darkorange",
    label="torch cholesky (dense)",
)
style_ax(ax2, "Cholesky factorization", "Time (ms)")

# Panel 3: triangular solve
ax3.plot(
    ns_all, t_spftrs, "o-", color="steelblue", label=f"curfp spftrs (RFP, {nrhs} rhs)"
)
ax3.plot(
    ns_torch_solv,
    t_torch_solv,
    "s--",
    color="tomato",
    label=f"torch.cholesky_solve ({nrhs} rhs)",
)
style_ax(ax3, "Triangular solve", "Time (ms)")

# Panel 4: memory (linear y-axis)
ax4.plot(ns_all, mem_rfp, "o-", color="steelblue", label="curfp (RFP)")
ax4.plot(ns_mem_dns, mem_dns, "s--", color="tomato", label="dense")
ax4.set_xlabel("Matrix size n")
ax4.set_ylabel("Memory (GB)")
ax4.set_title("Matrix memory")
ax4.set_xscale("log", base=2)
ax4.set_xticks(ns_all)
ax4.set_xticklabels([str(n) for n in ns_all], rotation=45, ha="right")
ax4.legend(fontsize=8)
ax4.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = "benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out}")
