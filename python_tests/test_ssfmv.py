"""
Tests for curfp.ssfmv — symmetric matrix-vector multiply in RFP format.

Validates all 8 RFP variants (transr N/T × uplo L/U × n even/odd) against
numpy matrix-vector multiply as reference.

Tests:
  - ssfmv vs numpy: y = alpha*A*x + beta*y
  - alpha/beta scaling
  - y accumulation (beta != 0)
  - Raw API smoke test
"""

import numpy as np
import pytest
import torch

import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS   = ["L", "U"]
N_VALS      = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128]

TOL = 1e-4


def make_sym_np(n: int, rng) -> np.ndarray:
    """Random n×n symmetric matrix."""
    A = rng.standard_normal((n, n)).astype(np.float32)
    return (A + A.T) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Core correctness: ssfmv vs numpy
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfmv_vs_numpy(n, uplo, transr):
    """curfp.ssfmv(y=0, beta=0): y = alpha*A*x matches numpy."""
    rng = np.random.default_rng(seed=11 + n)
    M_np = make_sym_np(n, rng)
    x_np = rng.standard_normal(n).astype(np.float32)
    alpha = float(rng.standard_normal())

    if uplo == "U":
        tri_np = np.triu(M_np)
    else:
        tri_np = np.tril(M_np)

    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)
    x_gpu = torch.from_numpy(x_np).cuda()

    y_gpu = curfp.ssfmv(arf, x_gpu, alpha=alpha, beta=0.0, transr=transr, uplo=uplo)
    y_np = y_gpu.cpu().numpy()

    y_ref = alpha * (M_np @ x_np)
    np.testing.assert_allclose(
        y_np, y_ref, rtol=TOL, atol=TOL,
        err_msg=f"ssfmv mismatch: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfmv_beta_accumulation(n, uplo, transr):
    """curfp.ssfmv with beta != 0: y = alpha*A*x + beta*y_in."""
    rng = np.random.default_rng(seed=23 + n)
    M_np = make_sym_np(n, rng)
    x_np = rng.standard_normal(n).astype(np.float32)
    y0_np = rng.standard_normal(n).astype(np.float32)
    alpha, beta = 2.5, -0.7

    if uplo == "U":
        tri_np = np.triu(M_np)
    else:
        tri_np = np.tril(M_np)

    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)
    x_gpu = torch.from_numpy(x_np).cuda()
    y_gpu = torch.from_numpy(y0_np.copy()).cuda()

    curfp.ssfmv(arf, x_gpu, y_gpu, alpha=alpha, beta=beta, transr=transr, uplo=uplo)
    y_np = y_gpu.cpu().numpy()

    y_ref = alpha * (M_np @ x_np) + beta * y0_np
    np.testing.assert_allclose(
        y_np, y_ref, rtol=TOL, atol=TOL,
        err_msg=f"ssfmv beta mismatch: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raw API smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_ssfmv_raw_api():
    """ssfmv_raw produces the same result as the high-level API."""
    n = 8
    rng = np.random.default_rng(42)
    M_np = make_sym_np(n, rng)
    x_np = rng.standard_normal(n).astype(np.float32)

    tri_np = np.triu(M_np)
    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr="T", uplo="U")
    x_gpu = torch.from_numpy(x_np).cuda()

    # High-level
    y_hl = curfp.ssfmv(arf, x_gpu, transr="T", uplo="U")

    # Raw
    y_raw = torch.zeros(n, dtype=torch.float32, device="cuda")
    with curfp.Handle() as h:
        curfp.ssfmv_raw(h, curfp.OP_T, curfp.FILL_UPPER, n,
                        1.0, arf, x_gpu, 1,
                        0.0, y_raw, 1)

    np.testing.assert_allclose(
        y_raw.cpu().numpy(), y_hl.cpu().numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="ssfmv_raw vs ssfmv mismatch"
    )
