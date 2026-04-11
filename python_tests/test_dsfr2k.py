"""Tests for curfp.dsfr2k — symmetric rank-2k update in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
# Only trans='T' is tested in the row-major convention (mirrors test_ssfr2k.py).
TRANS_VALS = ["T"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
K_VALS = [1, 4, 8]
TOL = 1e-9


def _make_sym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


@pytest.mark.parametrize("k", K_VALS)
@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("trans", TRANS_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfr2k_vs_numpy(transr, uplo, trans, n, k):
    """curfp.dsfr2k: C := alpha*(A*B^T + B*A^T) + beta*C matches numpy (float64, trans='T')."""
    rng = np.random.default_rng(seed=n * 1000 + k * 10 + ord(uplo) + ord(transr))
    alpha = float(rng.uniform(0.5, 2.0))
    beta = float(rng.uniform(0.0, 1.5))
    C_np = _make_sym(n, rng)

    # trans='T': A and B are row-major n×k, op(A)=A, op(B)=B → result n×n
    A_np = rng.standard_normal((n, k))
    B_np = rng.standard_normal((n, k))

    # Reference: C = alpha*(A @ B.T + B @ A.T) + beta*C
    C_ref = alpha * (A_np @ B_np.T + B_np @ A_np.T) + beta * C_np

    C_t = torch.from_numpy(C_np).cuda()
    arf = curfp.dstrttf(C_t, transr=transr, uplo=uplo)

    A_t = torch.from_numpy(np.ascontiguousarray(A_np)).cuda()
    B_t = torch.from_numpy(np.ascontiguousarray(B_np)).cuda()

    curfp.dsfr2k(
        A_t, B_t, arf, alpha=alpha, beta=beta, transr=transr, uplo=uplo, trans=trans
    )

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
        err_msg=f"dsfr2k mismatch: trans={trans} transr={transr} uplo={uplo} n={n} k={k}",
    )
