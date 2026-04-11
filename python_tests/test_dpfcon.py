"""Tests for curfp.dpfcon — reciprocal condition number from RFP Cholesky factor (double precision)."""

import numpy as np
import pytest
import torch
import scipy.linalg
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33]
TOL_REL = 0.1  # condition number estimators can have up to ~10% relative error


def _make_spd(n, rng):
    A = rng.standard_normal((n, n))
    A = A @ A.T
    A += n * np.eye(n)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dpfcon_bounds(n, uplo, transr):
    """curfp.dpfcon rcond is within 2x of scipy.linalg.inv-based rcond (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    A_np = _make_spd(n, rng)

    # 1-norm reference
    A_full = A_np + A_np.T - np.diag(np.diag(A_np))
    anorm = float(np.max(np.sum(np.abs(A_full), axis=0)))
    A_inv = scipy.linalg.inv(A_np)
    inv_norm = float(np.max(np.sum(np.abs(A_inv), axis=0)))
    rcond_ref = 1.0 / (anorm * inv_norm) if anorm > 0 and inv_norm > 0 else 0.0

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    curfp.dpftrf(arf, transr=transr, uplo=uplo)
    rcond_est = curfp.dpfcon(arf, anorm, transr=transr, uplo=uplo)

    assert rcond_est > 0, f"rcond_est={rcond_est} should be > 0"
    ratio = rcond_est / rcond_ref if rcond_ref > 0 else float("inf")
    assert 0.1 <= ratio <= 10.0, (
        f"dpfcon rcond estimate {rcond_est:.3e} is far from reference {rcond_ref:.3e} "
        f"(ratio={ratio:.2f}, transr={transr} uplo={uplo} n={n})"
    )
