"""
Tests for curfp.spfcon — reciprocal condition number estimator from RFP
Cholesky factor.

Validates all 4 transr/uplo combinations against scipy.linalg.lapack.spocon
(the dense SPD reference) for several matrix sizes and condition numbers.

Tests:
  - spfcon vs scipy spocon (all 4 RFP variants, multiple sizes)
  - Well-conditioned and moderately ill-conditioned matrices
  - n=1 edge case
  - Raw API smoke test
"""

import numpy as np
import pytest
import scipy.linalg
import scipy.linalg.lapack
import torch

import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS   = ["L", "U"]
N_VALS      = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]

# Condition estimators are inherently approximate; require agreement within
# a factor of 3 (i.e. ratio in [1/3, 3]).
FACTOR = 3.0


def make_spd(n: int, rng, cond: float = 10.0) -> np.ndarray:
    """Random n×n SPD matrix with approximate condition number `cond`."""
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)).astype(np.float64))
    # eigenvalues spaced from 1 to cond
    eigvals = np.linspace(1.0, cond, n)
    A = Q @ np.diag(eigvals) @ Q.T
    return A.astype(np.float32)


def scipy_rcond(A_np: np.ndarray) -> float:
    """Reference rcond via scipy's dense SPOCON."""
    A64 = A_np.astype(np.float64)
    c_upper = scipy.linalg.cholesky(A64, lower=False)          # upper Cholesky
    anorm = float(scipy.linalg.norm(A64, 1))
    rcond_ref, info = scipy.linalg.lapack.spocon(c_upper, anorm)
    assert info == 0, f"scipy spocon failed: info={info}"
    return float(rcond_ref)


def curfp_rcond(A_np: np.ndarray, transr: str, uplo: str) -> float:
    """curfp spfcon for the given RFP variant."""
    n = A_np.shape[0]
    if uplo == "U":
        tri_np = np.triu(A_np)
    else:
        tri_np = np.tril(A_np)
    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)
    curfp.spftrf(arf, transr=transr, uplo=uplo)
    anorm = float(np.linalg.norm(A_np, 1))
    return curfp.spfcon(arf, anorm, transr=transr, uplo=uplo)


# ─────────────────────────────────────────────────────────────────────────────
# Core correctness: spfcon vs scipy spocon
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_spfcon_well_conditioned(n, uplo, transr):
    """spfcon matches scipy spocon for well-conditioned matrices (cond≈10)."""
    rng = np.random.default_rng(seed=42 + n)
    A_np = make_spd(n, rng, cond=10.0)
    rcond_ref = scipy_rcond(A_np)
    rcond_curfp = curfp_rcond(A_np, transr, uplo)
    ratio = rcond_curfp / rcond_ref if rcond_ref > 0 else float("inf")
    assert 1.0 / FACTOR <= ratio <= FACTOR, (
        f"n={n}, transr={transr!r}, uplo={uplo!r}: "
        f"rcond_curfp={rcond_curfp:.4e}, rcond_ref={rcond_ref:.4e}, ratio={ratio:.3f}"
    )


@pytest.mark.parametrize("n",      [8, 32, 64])
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_spfcon_ill_conditioned(n, uplo, transr):
    """spfcon matches scipy spocon for moderately ill-conditioned matrices."""
    rng = np.random.default_rng(seed=99 + n)
    A_np = make_spd(n, rng, cond=1e4)
    rcond_ref = scipy_rcond(A_np)
    rcond_curfp = curfp_rcond(A_np, transr, uplo)
    ratio = rcond_curfp / rcond_ref if rcond_ref > 0 else float("inf")
    assert 1.0 / FACTOR <= ratio <= FACTOR, (
        f"n={n}, transr={transr!r}, uplo={uplo!r}: "
        f"rcond_curfp={rcond_curfp:.4e}, rcond_ref={rcond_ref:.4e}, ratio={ratio:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_spfcon_n1():
    """n=1 edge case: A = [[v]], any 1×1 SPD matrix has rcond=1."""
    v = 5.0  # the matrix A = [[v]]
    arf = torch.tensor([v], dtype=torch.float32, device="cuda")
    curfp.spftrf(arf, transr="T", uplo="U")  # arf[0] = sqrt(v)
    anorm = v                                # 1-norm of [[v]] = v
    rcond = curfp.spfcon(arf, anorm, transr="T", uplo="U")
    # A = [[v]], A^{-1} = [[1/v]]; rcond = 1 / (||A^{-1}||_1 * anorm)
    #                                    = 1 / ((1/v) * v) = 1.0
    assert abs(rcond - 1.0) < 0.01, f"n=1 rcond={rcond}, expected 1.0"


def test_spfcon_anorm_zero():
    """anorm=0 → rcond=0 (no division)."""
    arf = torch.ones(1, dtype=torch.float32, device="cuda")
    rcond = curfp.spfcon(arf, 0.0, transr="T", uplo="U")
    assert rcond == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Raw API smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_spfcon_raw_api():
    """spfcon_raw produces the same result as the high-level API."""
    n = 16
    rng = np.random.default_rng(7)
    A_np = make_spd(n, rng, cond=50.0)
    tri_np = np.triu(A_np)
    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr="T", uplo="U")
    curfp.spftrf(arf, transr="T", uplo="U")
    anorm = float(np.linalg.norm(A_np, 1))

    # High-level
    rcond_hl = curfp.spfcon(arf, anorm, transr="T", uplo="U")

    # Raw
    with curfp.Handle() as h:
        rcond_raw = curfp.spfcon_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, arf, anorm)

    assert abs(rcond_hl - rcond_raw) < 1e-7, (
        f"spfcon_raw vs spfcon mismatch: {rcond_raw} vs {rcond_hl}"
    )
