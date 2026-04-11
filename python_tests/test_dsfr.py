"""Tests for curfp.dsfr — symmetric rank-1 update in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128]
TOL = 1e-10


def _make_sym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfr_vs_numpy(n, uplo, transr):
    """curfp.dsfr: arf += alpha*x*x^T matches numpy (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    alpha = float(rng.uniform(0.5, 2.5))
    A_np = _make_sym(n, rng)
    x_np = rng.standard_normal((n,))

    A_ref = A_np + alpha * np.outer(x_np, x_np)

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)

    x_t = torch.from_numpy(x_np).cuda()
    curfp.dsfr(arf, x_t, alpha=alpha, transr=transr, uplo=uplo)

    A_out = curfp.dstfttr(arf, transr=transr, uplo=uplo)
    A_out_np = A_out.cpu().numpy()

    if uplo == "L":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        mask = np.triu(np.ones((n, n), dtype=bool))

    np.testing.assert_allclose(
        A_out_np[mask],
        A_ref[mask],
        rtol=TOL,
        atol=TOL,
        err_msg=f"dsfr mismatch: transr={transr} uplo={uplo} n={n}",
    )


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfr_zero_alpha(n, uplo, transr):
    """alpha=0: arf must be unchanged."""
    rng = np.random.default_rng(seed=n * 7 + ord(uplo))
    A_np = _make_sym(n, rng)
    x_np = rng.standard_normal((n,))

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    arf0 = arf.clone()

    x_t = torch.from_numpy(x_np).cuda()
    curfp.dsfr(arf, x_t, alpha=0.0, transr=transr, uplo=uplo)

    assert torch.allclose(arf, arf0), (
        f"alpha=0: arf changed for transr={transr} uplo={uplo} n={n}"
    )


def test_dsfr_raw_api():
    """dsfr_raw produces the same result as the high-level API."""
    n = 8
    rng = np.random.default_rng(42)
    A_np = _make_sym(n, rng)
    x_np = rng.standard_normal((n,))
    alpha = 1.5

    A_t = torch.from_numpy(A_np).cuda()
    x_t = torch.from_numpy(x_np).cuda()

    arf1 = curfp.dstrttf(A_t)
    arf2 = arf1.clone()

    curfp.dsfr(arf1, x_t, alpha=alpha)
    curfp.dsfr_raw(curfp.Handle(), curfp.OP_T, curfp.FILL_UPPER, n, alpha, x_t, 1, arf2)

    assert torch.allclose(arf1, arf2, rtol=1e-12, atol=1e-12), (
        "dsfr_raw differs from high-level dsfr"
    )
