"""Tests for curfp.dsfmm — symmetric matrix-matrix multiply in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
SIDE_VALS = ["L", "R"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
NRHS_VALS = [1, 4, 8]
TOL = 1e-9


def _make_sym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


def _dsfmm_call(h, transr, uplo, side, n_A, nrhs, alpha, arf, B_np, beta, C_np):
    """Call dsfmm_raw with the row-major convention and return updated C (nrhs×n_A)."""
    if side == "L":
        B_t = torch.from_numpy(np.ascontiguousarray(B_np)).cuda()
        C_t = torch.from_numpy(np.ascontiguousarray(C_np)).cuda()
        curfp.dsfmm_raw(
            h,
            curfp._op(transr),
            curfp._fill(uplo),
            curfp.SIDE_LEFT,
            n_A,
            nrhs,
            alpha,
            arf,
            B_t,
            n_A,
            beta,
            C_t,
            n_A,
        )
        return C_t.cpu().numpy()
    else:
        B_T = np.ascontiguousarray(B_np.T)
        C_T = np.ascontiguousarray(C_np.T)
        B_T_t = torch.from_numpy(B_T).cuda()
        C_T_t = torch.from_numpy(C_T).cuda()
        curfp.dsfmm_raw(
            h,
            curfp._op(transr),
            curfp._fill(uplo),
            curfp.SIDE_RIGHT,
            nrhs,
            n_A,
            alpha,
            arf,
            B_T_t,
            nrhs,
            beta,
            C_T_t,
            nrhs,
        )
        return C_T_t.cpu().numpy().T


@pytest.mark.parametrize("nrhs", NRHS_VALS)
@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("side", SIDE_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_dsfmm_vs_numpy(transr, uplo, side, n, nrhs):
    """curfp.dsfmm: C := alpha * A * B + beta * C matches numpy (float64)."""
    rng = np.random.default_rng(
        seed=n * 100 + nrhs * 11 + ord(uplo) + ord(transr) + ord(side)
    )
    alpha = float(rng.uniform(0.5, 2.0))
    beta = float(rng.uniform(0.1, 1.5))
    A_np = _make_sym(n, rng)
    B_np = rng.standard_normal((nrhs, n))
    C_np = rng.standard_normal((nrhs, n))

    C_ref = alpha * (B_np @ A_np) + beta * C_np

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)

    with curfp.Handle() as h:
        C_out = _dsfmm_call(
            h, transr, uplo, side, n, nrhs, alpha, arf, B_np, beta, C_np
        )

    np.testing.assert_allclose(
        C_out,
        C_ref,
        rtol=TOL,
        atol=TOL,
        err_msg=f"dsfmm mismatch: transr={transr} uplo={uplo} side={side} n={n} nrhs={nrhs}",
    )
