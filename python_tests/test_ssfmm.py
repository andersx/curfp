"""
Tests for curfp.ssfmm — symmetric matrix-matrix multiply in RFP format.

Row-major (PyTorch/Python) calling conventions
-----------------------------------------------
cuBLAS ssymm operates on column-major matrices.  For Python row-major (C-contiguous)
tensors, the standard transpose trick applies:

side='L', m=n_A, n=nrhs, ldb=n_A:
    cuBLAS sees B as n_A×nrhs col-major (= B_python.T), and computes
        C_cuBLAS = A * B_python.T  (n_A×n_A) * (n_A×nrhs) = n_A×nrhs col-major
    which maps back to Python as
        C_python = (A * B.T).T = B @ A    [since A symmetric]
    Reference:  C = alpha * B @ A + beta * C

side='R', m=nrhs, n=n_A, ldb=nrhs, with B.T and C.T passed in:
    Same mathematical result; B and C must be pre/post-transposed.
    Reference:  C.T = alpha * A @ B.T + beta * C.T

Both sides are tested using a common helper that handles the transposition.
"""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
SIDE_VALS = ["L", "R"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
NRHS_VALS = [1, 4, 8]
TOL = 1e-3


def _make_sym(n, rng):
    A = rng.standard_normal((n, n)).astype(np.float32)
    return (A + A.T) / 2.0


def _ssfmm_call(h, transr, uplo, side, n_A, nrhs, alpha, arf, B_np, beta, C_np):
    """Call ssfmm_raw with the row-major convention and return updated C (nrhs×n_A).

    Both sides compute:  C[i,:] = alpha*(A @ B[i,:]) + beta*C[i,:]

    side='L': pass B directly (nrhs×n_A), m=n_A, n=nrhs, ldb=n_A.
    side='R': pass B.T (n_A×nrhs) and C.T, result transposed back.
    """
    if side == "L":
        B_t = torch.from_numpy(np.ascontiguousarray(B_np)).cuda()
        C_t = torch.from_numpy(np.ascontiguousarray(C_np)).cuda()
        curfp.ssfmm_raw(
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
        # side='R': B.T has shape (n_A, nrhs); ldb=nrhs (≥ m=nrhs ✓)
        B_T = np.ascontiguousarray(B_np.T)  # (n_A, nrhs)
        C_T = np.ascontiguousarray(C_np.T)  # (n_A, nrhs)
        B_T_t = torch.from_numpy(B_T).cuda()
        C_T_t = torch.from_numpy(C_T).cuda()
        curfp.ssfmm_raw(
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
        return C_T_t.cpu().numpy().T  # (nrhs, n_A)


@pytest.mark.parametrize("nrhs", NRHS_VALS)
@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("side", SIDE_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfmm_vs_numpy(transr, uplo, side, n, nrhs):
    """curfp.ssfmm matches numpy reference C = alpha*B@A + beta*C."""
    rng = np.random.default_rng(seed=n * 1000 + nrhs * 10 + ord(side) + ord(uplo))
    alpha = float(rng.uniform(0.5, 2.0))
    beta = float(rng.uniform(0.0, 1.5))

    A_np = _make_sym(n, rng)
    B_np = rng.standard_normal((nrhs, n)).astype(np.float32)
    C_np = rng.standard_normal((nrhs, n)).astype(np.float32)

    # Reference: C[i,:] = alpha*(A @ B[i,:]) + beta*C[i,:]
    C_ref = alpha * (B_np @ A_np) + beta * C_np

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.strttf(A_t, transr=transr, uplo=uplo)

    h = curfp.Handle()
    C_out = _ssfmm_call(h, transr, uplo, side, n, nrhs, alpha, arf, B_np, beta, C_np)

    np.testing.assert_allclose(
        C_out,
        C_ref,
        rtol=TOL,
        atol=TOL,
        err_msg=f"ssfmm mismatch: transr={transr} uplo={uplo} side={side} n={n} nrhs={nrhs}",
    )


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("side", SIDE_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfmm_zero_alpha(transr, uplo, side, n):
    """alpha=0, beta=1: C unchanged."""
    nrhs = 4
    rng = np.random.default_rng(555)
    A_np = _make_sym(n, rng)
    B_np = rng.standard_normal((nrhs, n)).astype(np.float32)
    C_np = rng.standard_normal((nrhs, n)).astype(np.float32)
    C0 = C_np.copy()

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.strttf(A_t, transr=transr, uplo=uplo)
    h = curfp.Handle()

    C_out = _ssfmm_call(h, transr, uplo, side, n, nrhs, 0.0, arf, B_np, 1.0, C_np)

    np.testing.assert_allclose(
        C_out,
        C0,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"alpha=0, beta=1: C changed for transr={transr} uplo={uplo} side={side} n={n}",
    )


if __name__ == "__main__":
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 15, 16]:
        for transr in ["N", "T"]:
            for uplo in ["L", "U"]:
                for side in ["L", "R"]:
                    for nrhs in [1, 4]:
                        test_ssfmm_vs_numpy(transr, uplo, side, n, nrhs)
    print("All ssfmm tests PASSED!")
