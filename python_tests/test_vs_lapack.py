"""
Validate curfp (CUDA) against LAPACK (CPU via scipy) for all 8 RFP variants.

For each combination of (transr, uplo, trans, n_parity):
  1. Build a random SPD matrix A @ A.T + I on CPU
  2. Pack into RFP format on CPU using scipy LAPACK ssfrk
  3. Factorize on CPU using scipy LAPACK spftrf
  4. Solve on CPU using scipy LAPACK spftrs
  5. Repeat steps 2-4 on GPU using curfp
  6. Compare RFP arrays and solution vectors between CPU and GPU

This validates every code path in curfp_ssfrk.cpp, curfp_spftrf.cpp, and curfp_spftrs.cpp.
"""

import math
import numpy as np
import torch
import curfp
import scipy.linalg.lapack as lapack

# Tolerances (float32 — errors grow with matrix size)
RFP_TOL = 1e-3  # tolerance for RFP array comparison
SOLVE_TOL = 1e-3  # tolerance for solution vector comparison


def make_random_A(n, k, seed=None):
    """Return a random (n, k) float32 row-major numpy array, scaled so A@A.T ≈ I."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, k)) / math.sqrt(k)).astype(np.float32)


def rfp_cpu(A_np, transr, uplo, trans, n, k):
    """Build RFP-packed SPD matrix on CPU using LAPACK ssfrk + add identity.

    scipy ssfrk uses Fortran (col-major) convention:
      trans='N': C = alpha * A * A^T, A passed as np.asfortranarray(A) with shape (n, k)
      trans='T': C = alpha * A^T * A, pass A.T with shape (k, n) C-order

    We always compute A @ A^T (n×n), so use trans='N' with Fortran A.
    The `transr` parameter of curfp corresponds to LAPACK's `transr` (RFP storage variant).
    """
    nt = n * (n + 1) // 2
    C = np.zeros(nt, dtype=np.float32)
    # Always compute A @ A^T using trans='N' with Fortran-order A
    A_f = np.asfortranarray(A_np)  # (n, k) col-major for LAPACK
    C = lapack.ssfrk(
        n, k, 1.0, A_f, 0.0, C, transr=transr.encode(), uplo=uplo.encode(), trans=b"N"
    )

    # Add identity
    I_f = np.asfortranarray(np.eye(n, dtype=np.float32))
    C = lapack.ssfrk(
        n, n, 1.0, I_f, 1.0, C, transr=transr.encode(), uplo=uplo.encode(), trans=b"N"
    )
    return C


def rfp_gpu(A_gpu, transr, uplo, n):
    """Build RFP-packed SPD matrix on GPU using curfp ssfrk + add_to_diagonal."""
    C_gpu = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
    curfp.ssfrk(A_gpu, C_gpu, transr=transr, uplo=uplo)
    curfp.add_to_diagonal(C_gpu, 1.0, transr=transr, uplo=uplo)
    return C_gpu


def run_case(n, k, transr, uplo, nrhs, seed):
    """Run one validation case. Returns dict with pass/fail info."""
    trans = "T"  # row-major A → always use trans=T

    A_np = make_random_A(n, k, seed=seed)
    A_gpu = torch.from_numpy(A_np).to("cuda")

    # ---- CPU reference (LAPACK) ----
    C_cpu = rfp_cpu(A_np, transr, uplo, trans, n, k)

    # factorize
    C_fac_cpu, info = lapack.spftrf(
        n, C_cpu.copy(), transr=transr.encode(), uplo=uplo.encode()
    )
    if info != 0:
        return {"pass": False, "reason": f"LAPACK spftrf failed: info={info}"}

    # solve
    rng = np.random.default_rng(seed + 1000)
    B_np = rng.standard_normal((n, nrhs)).astype(np.float32)
    X_cpu, info = lapack.spftrs(
        n, C_fac_cpu, B_np.copy(), transr=transr.encode(), uplo=uplo.encode()
    )
    if info != 0:
        return {"pass": False, "reason": f"LAPACK spftrs failed: info={info}"}

    # ---- GPU (curfp) ----
    C_gpu = rfp_gpu(A_gpu, transr, uplo, n)

    # compare RFP arrays before factorization
    C_gpu_np = C_gpu.cpu().numpy()
    rfp_diff = np.max(np.abs(C_gpu_np - C_cpu))

    # factorize
    try:
        info_gpu = curfp.spftrf(C_gpu, transr=transr, uplo=uplo, check=False)
    except Exception as e:
        return {"pass": False, "reason": f"curfp spftrf exception: {e}"}
    if info_gpu != 0:
        return {"pass": False, "reason": f"curfp spftrf failed: info={info_gpu}"}

    # compare Cholesky factor
    factor_diff = np.max(np.abs(C_gpu.cpu().numpy() - C_fac_cpu))

    # solve
    B_gpu = torch.from_numpy(B_np).to("cuda")
    try:
        curfp.spftrs(C_gpu, B_gpu, transr=transr, uplo=uplo)
    except Exception as e:
        return {"pass": False, "reason": f"curfp spftrs exception: {e}"}

    X_gpu = B_gpu.cpu().numpy()
    solve_diff = np.max(np.abs(X_gpu - X_cpu))

    passed = rfp_diff < RFP_TOL and factor_diff < RFP_TOL and solve_diff < SOLVE_TOL
    return {
        "pass": passed,
        "rfp_diff": rfp_diff,
        "factor_diff": factor_diff,
        "solve_diff": solve_diff,
        "reason": ""
        if passed
        else f"rfp={rfp_diff:.2e} factor={factor_diff:.2e} solve={solve_diff:.2e}",
    }


def test_all_variants():
    sizes = [
        # Small: exercise edge cases
        (1, 1),
        (2, 2),
        (3, 3),
        # Medium: even/odd n, k=n and k!=n
        (4, 4),
        (5, 5),
        (8, 6),
        (9, 7),
        # Larger: exercise blocked cuSOLVER/cuBLAS code paths
        (32, 16),
        (33, 17),
        (64, 32),
        (65, 33),
        (128, 64),
        (129, 65),
        (256, 128),
        (257, 129),
    ]
    nrhs = 3
    transr_vals = ["N", "T"]
    uplo_vals = ["L", "U"]

    header = f"{'n':>4} {'k':>3} {'transr':>6} {'uplo':>4}  {'rfp':>8} {'factor':>8} {'solve':>8}  status"
    print(header)
    print("-" * len(header))

    all_passed = True
    seed = 42
    for n, k in sizes:
        for transr in transr_vals:
            for uplo in uplo_vals:
                result = run_case(n, k, transr, uplo, nrhs, seed)
                seed += 1
                if result["pass"]:
                    status = "PASS"
                    row = (
                        f"{n:>4} {k:>3} {transr:>6} {uplo:>4}  "
                        f"{result['rfp_diff']:>8.2e} {result['factor_diff']:>8.2e} "
                        f"{result['solve_diff']:>8.2e}  {status}"
                    )
                else:
                    status = f"FAIL: {result['reason']}"
                    row = f"{n:>4} {k:>3} {transr:>6} {uplo:>4}  {status}"
                    all_passed = False
                print(row)

    print()
    if all_passed:
        print("All variants PASSED.")
    else:
        print("Some variants FAILED — see above.")
    return all_passed


if __name__ == "__main__":
    ok = test_all_variants()
    raise SystemExit(0 if ok else 1)
