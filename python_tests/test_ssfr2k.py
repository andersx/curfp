"""
Tests for curfp.ssfr2k — symmetric rank-2k update in RFP format.

Validates all 8 RFP variants (transr N/T × uplo L/U × n even/odd) against
the numpy reference:
    C = alpha * (op(A) @ op(B).T + op(B) @ op(A).T) + beta * C
"""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
# trans='N' is designed for F-order (col-major) input; the standard row-major
# Python convention uses trans='T'.  Only parametrize over trans='T' here.
TRANS_VALS = ["T"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
K_VALS = [1, 4, 8]
TOL = 1e-3


def _make_sym(n, rng):
    A = rng.standard_normal((n, n)).astype(np.float32)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("k", K_VALS)
@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("trans", TRANS_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfr2k_vs_numpy(transr, uplo, trans, n, k):
    """curfp.ssfr2k matches numpy reference."""
    rng = np.random.default_rng(seed=n * 1000 + k * 10 + ord(uplo) + ord(transr))
    alpha = float(rng.uniform(0.5, 2.0))
    beta = float(rng.uniform(0.0, 1.5))

    C_np = _make_sym(n, rng)

    # trans='T': A and B are row-major n×k, op(A)=A, op(B)=B → result n×n
    A_np = rng.standard_normal((n, k)).astype(np.float32)
    B_np = rng.standard_normal((n, k)).astype(np.float32)

    # Reference: C = alpha*(A @ B.T + B @ A.T) + beta*C
    C_ref = alpha * (A_np @ B_np.T + B_np @ A_np.T) + beta * C_np

    # Pack C to RFP
    C_t = torch.from_numpy(C_np).cuda()
    arf = curfp.strttf(C_t, transr=transr, uplo=uplo)

    A_t = torch.from_numpy(A_np).cuda()
    B_t = torch.from_numpy(B_np).cuda()

    curfp.ssfr2k(
        A_t, B_t, arf, alpha=alpha, beta=beta, transr=transr, uplo=uplo, trans=trans
    )

    # Unpack
    C_out = curfp.stfttr(arf, transr=transr, uplo=uplo)
    C_out_np = C_out.cpu().numpy()

    # Compare stored triangle
    if uplo == "L":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        mask = np.triu(np.ones((n, n), dtype=bool))

    np.testing.assert_allclose(
        C_out_np[mask],
        C_ref[mask],
        rtol=TOL,
        atol=TOL,
        err_msg=f"ssfr2k mismatch: transr={transr} uplo={uplo} trans={trans} n={n} k={k}",
    )


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfr2k_passthrough(transr, uplo, n):
    """alpha=0, beta=1: C unchanged."""
    rng = np.random.default_rng(42)
    C_np = _make_sym(n, rng)
    # trans='T': A and B are (n, 4) row-major
    A_np = rng.standard_normal((n, 4)).astype(np.float32)
    B_np = rng.standard_normal((n, 4)).astype(np.float32)

    C_t = torch.from_numpy(C_np).cuda()
    arf = curfp.strttf(C_t, transr=transr, uplo=uplo)
    arf0 = arf.clone()

    A_t = torch.from_numpy(A_np).cuda()
    B_t = torch.from_numpy(B_np).cuda()
    curfp.ssfr2k(
        A_t, B_t, arf, alpha=0.0, beta=1.0, transr=transr, uplo=uplo, trans="T"
    )

    assert torch.allclose(arf, arf0), (
        f"alpha=0, beta=1: arf changed for transr={transr} uplo={uplo} n={n}"
    )


def test_ssfr2k_raw_api():
    """ssfr2k_raw produces the same result as the high-level API."""
    n, k = 8, 4
    rng = np.random.default_rng(77)
    C_np = _make_sym(n, rng)
    # trans='T': A and B are (n, k) row-major; k_rank=k=A.shape[1]; lda=k
    A_np = rng.standard_normal((n, k)).astype(np.float32)
    B_np = rng.standard_normal((n, k)).astype(np.float32)
    alpha, beta = 1.3, 0.7

    C_t = torch.from_numpy(C_np).cuda()
    A_t = torch.from_numpy(A_np).cuda()
    B_t = torch.from_numpy(B_np).cuda()

    arf1 = curfp.strttf(C_t)
    arf2 = arf1.clone()

    curfp.ssfr2k(A_t, B_t, arf1, alpha=alpha, beta=beta)
    curfp.ssfr2k_raw(
        curfp.Handle(),
        curfp.OP_T,
        curfp.FILL_UPPER,
        curfp.OP_T,
        n,
        k,
        alpha,
        A_t,
        k,
        B_t,
        k,
        beta,
        arf2,
    )

    assert torch.allclose(arf1, arf2, rtol=1e-5, atol=1e-5), (
        "ssfr2k_raw result differs from high-level ssfr2k"
    )


if __name__ == "__main__":
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 15, 16]:
        for transr in ["N", "T"]:
            for uplo in ["L", "U"]:
                for trans in ["T"]:
                    for k in [1, 4]:
                        test_ssfr2k_vs_numpy(transr, uplo, trans, n, k)
    test_ssfr2k_raw_api()
    print("All ssfr2k tests PASSED!")
