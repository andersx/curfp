"""Tests for curfp.dsfmv — symmetric matrix-vector multiply in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128]
TOL = 1e-10


def _make_spd(n, rng):
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfmv_vs_numpy(n, uplo, transr):
    """curfp.dsfmv: y = alpha*A*x + beta*y matches numpy (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    alpha = float(rng.uniform(0.5, 2.5))
    beta = float(rng.uniform(0.1, 1.5))
    A_np = _make_spd(n, rng)
    x_np = rng.standard_normal((n,))
    y_np = rng.standard_normal((n,))

    y_ref = alpha * A_np @ x_np + beta * y_np

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)

    x_t = torch.from_numpy(x_np).cuda()
    y_t = torch.from_numpy(y_np.copy()).cuda()
    curfp.dsfmv(arf, x_t, y_t, alpha=alpha, beta=beta, transr=transr, uplo=uplo)

    np.testing.assert_allclose(
        y_t.cpu().numpy(),
        y_ref,
        rtol=TOL,
        atol=TOL,
        err_msg=f"dsfmv mismatch: transr={transr} uplo={uplo} n={n}",
    )


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfmv_zero_alpha(n, uplo, transr):
    """alpha=0, beta=1: y unchanged."""
    rng = np.random.default_rng(seed=n * 7 + ord(uplo))
    A_np = _make_spd(n, rng)
    x_np = rng.standard_normal((n,))
    y_np = rng.standard_normal((n,))

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    x_t = torch.from_numpy(x_np).cuda()
    y_t = torch.from_numpy(y_np.copy()).cuda()
    y0 = y_t.clone()

    curfp.dsfmv(arf, x_t, y_t, alpha=0.0, beta=1.0, transr=transr, uplo=uplo)

    assert torch.allclose(y_t, y0, rtol=1e-12, atol=1e-12)
