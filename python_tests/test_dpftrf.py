"""Tests for curfp.dpftrf — Cholesky factorization in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import scipy.linalg
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
TOL = 1e-10


def _make_spd(n, rng):
    A = rng.standard_normal((n, n))
    A = A @ A.T
    A += n * np.eye(n)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dpftrf_vs_scipy(n, uplo, transr):
    """curfp.dpftrf Cholesky factor matches scipy (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    A_np = _make_spd(n, rng)

    # Reference Cholesky
    L_ref = scipy.linalg.cholesky(A_np, lower=True)

    # RFP Cholesky
    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    info = curfp.dpftrf(arf, transr=transr, uplo=uplo, check=True)
    assert info == 0

    # Unpack factor to full matrix
    L_curfp = curfp.dstfttr(arf, transr=transr, uplo=uplo).cpu().numpy()
    if uplo == "U":
        # stfttr gives U; reference is L
        L_curfp = L_curfp.T

    # Compare lower triangle only
    mask = np.tril(np.ones((n, n), dtype=bool))
    np.testing.assert_allclose(
        L_curfp[mask],
        L_ref[mask],
        rtol=TOL,
        atol=TOL,
        err_msg=f"dpftrf mismatch: transr={transr} uplo={uplo} n={n}",
    )


def test_dpftrf_not_pd():
    """dpftrf returns info > 0 for non-positive-definite matrix when check=False."""
    n = 4
    A = torch.tensor(
        [-1.0] + [0.0] * (n * (n + 1) // 2 - 1), dtype=torch.float64, device="cuda"
    )
    info = curfp.dpftrf(A, check=False)
    assert info > 0, f"Expected info>0 for non-PD matrix, got {info}"
