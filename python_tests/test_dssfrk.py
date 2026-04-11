"""Tests for curfp.dsfrk — symmetric rank-k update in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
# n=1 is skipped: ssyrk with sub-block dim=0 is undefined in cuBLAS.
N_VALS = [2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
K_VALS = [1, 4, 8]
TOL = 1e-9


def _make_sym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


@pytest.mark.parametrize("k", K_VALS)
@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfrk_vs_numpy(transr, uplo, n, k):
    """curfp.dsfrk: C := alpha*A*A^T + beta*C matches numpy (float64).

    Uses row-major convention (trans='T'): A is (n, k), lda=k (auto-inferred).
    Reference: C_ref = alpha * (A @ A.T) + beta * C
    """
    rng = np.random.default_rng(seed=n * 100 + k * 13 + ord(uplo) + ord(transr))
    alpha = float(rng.uniform(0.5, 2.5))
    beta = float(rng.uniform(0.1, 1.5))
    C_np = _make_sym(n, rng)
    A_np = rng.standard_normal((n, k))  # trans='T' row-major: (n, k)

    # Reference: C = alpha * A @ A.T + beta * C
    C_ref = alpha * (A_np @ A_np.T) + beta * C_np

    C_t = torch.from_numpy(C_np).cuda()
    arf = curfp.dstrttf(C_t, transr=transr, uplo=uplo)

    A_t = torch.from_numpy(np.ascontiguousarray(A_np)).cuda()

    # trans='T', A is (n,k): lda auto-inferred as A.shape[1]=k
    curfp.dsfrk(A_t, arf, alpha=alpha, beta=beta, transr=transr, uplo=uplo, trans="T")

    A_out = curfp.dstfttr(arf, transr=transr, uplo=uplo)
    A_out_np = A_out.cpu().numpy()

    if uplo == "L":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        mask = np.triu(np.ones((n, n), dtype=bool))

    np.testing.assert_allclose(
        A_out_np[mask],
        C_ref[mask],
        rtol=TOL,
        atol=TOL,
        err_msg=f"dsfrk mismatch: transr={transr} uplo={uplo} n={n} k={k}",
    )


def test_dsfrk_passthrough():
    """alpha=0, beta=1: C unchanged."""
    n, k = 6, 3
    rng = np.random.default_rng(0)
    C_np = _make_sym(n, rng)
    A_np = rng.standard_normal((n, k))

    C_t = torch.from_numpy(C_np).cuda()
    arf = curfp.dstrttf(C_t)
    arf0 = arf.clone()

    A_t = torch.from_numpy(np.ascontiguousarray(A_np)).cuda()
    curfp.dsfrk(A_t, arf, alpha=0.0, beta=1.0, trans="T")

    assert torch.allclose(arf, arf0, rtol=1e-12, atol=1e-12)
