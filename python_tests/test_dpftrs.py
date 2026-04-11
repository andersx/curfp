"""Tests for curfp.dpftrs — triangular solve using RFP Cholesky factor (double precision)."""

import numpy as np
import pytest
import torch
import scipy.linalg
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
NRHS_VALS = [1, 4, 8]
TOL = 1e-8


def _make_spd(n, rng):
    A = rng.standard_normal((n, n))
    A = A @ A.T
    A += n * np.eye(n)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("nrhs", NRHS_VALS)
@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dpftrs_vs_numpy(n, uplo, transr, nrhs):
    """curfp.dpftrs solve A*X=B matches numpy (float64)."""
    rng = np.random.default_rng(seed=n * 100 + nrhs * 11 + ord(uplo) + ord(transr))
    A_np = _make_spd(n, rng)
    B_np = rng.standard_normal((n, nrhs))

    X_ref = scipy.linalg.solve(A_np, B_np, assume_a="pos")

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    curfp.dpftrf(arf, transr=transr, uplo=uplo)

    B_t = torch.from_numpy(B_np.copy()).cuda()
    curfp.dpftrs(arf, B_t, transr=transr, uplo=uplo)

    np.testing.assert_allclose(
        B_t.cpu().numpy(),
        X_ref,
        rtol=TOL,
        atol=TOL,
        err_msg=f"dpftrs mismatch: transr={transr} uplo={uplo} n={n} nrhs={nrhs}",
    )
