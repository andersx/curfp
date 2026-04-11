"""Tests for curfp.dpftri — SPD matrix inverse from RFP Cholesky factor (double precision)."""

import numpy as np
import pytest
import torch
import scipy.linalg
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
TOL = 1e-8


def _make_spd(n, rng):
    A = rng.standard_normal((n, n))
    A = A @ A.T
    A += n * np.eye(n)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dpftri_vs_numpy(n, uplo, transr):
    """curfp.dpftri: inverse matches scipy.linalg.inv (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    A_np = _make_spd(n, rng)

    A_inv_ref = scipy.linalg.inv(A_np)

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    curfp.dpftrf(arf, transr=transr, uplo=uplo)
    curfp.dpftri(arf, transr=transr, uplo=uplo)

    A_inv = curfp.dstfttr(arf, transr=transr, uplo=uplo).cpu().numpy()
    # stfttr only fills one triangle; symmetrise for comparison
    A_inv_full = A_inv + A_inv.T
    np.fill_diagonal(A_inv_full, np.diag(A_inv))

    np.testing.assert_allclose(
        A_inv_full,
        A_inv_ref,
        rtol=TOL,
        atol=TOL,
        err_msg=f"dpftri mismatch: transr={transr} uplo={uplo} n={n}",
    )
