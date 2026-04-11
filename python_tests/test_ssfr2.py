"""
Tests for curfp.ssfr2 — symmetric rank-2 update in RFP format.

Validates all 8 RFP variants (transr N/T × uplo L/U × n even/odd) by:
  1. Building a random symmetric matrix, packing to RFP.
  2. Calling ssfr2 with random vectors x and y.
  3. Comparing result (via stfttr) to numpy reference:
       A += alpha * (outer(x,y) + outer(y,x))
"""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128]
TOL = 1e-4


def _make_sym(n, rng):
    A = rng.standard_normal((n, n)).astype(np.float32)
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfr2_vs_numpy(n, uplo, transr):
    """curfp.ssfr2: arf += alpha*(x*y^T + y*x^T) matches numpy."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    alpha = float(rng.uniform(0.5, 2.5))
    A_np = _make_sym(n, rng)
    x_np = rng.standard_normal((n,)).astype(np.float32)
    y_np = rng.standard_normal((n,)).astype(np.float32)

    # Reference
    A_ref = A_np + alpha * (np.outer(x_np, y_np) + np.outer(y_np, x_np))

    # Pack
    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.strttf(A_t, transr=transr, uplo=uplo)

    # Apply ssfr2
    x_t = torch.from_numpy(x_np).cuda()
    y_t = torch.from_numpy(y_np).cuda()
    curfp.ssfr2(arf, x_t, y_t, alpha=alpha, transr=transr, uplo=uplo)

    # Unpack
    A_out = curfp.stfttr(arf, transr=transr, uplo=uplo)
    A_out_np = A_out.cpu().numpy()

    # Compare stored triangle
    if uplo == "L":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        mask = np.triu(np.ones((n, n), dtype=bool))

    np.testing.assert_allclose(
        A_out_np[mask],
        A_ref[mask],
        rtol=TOL,
        atol=TOL,
        err_msg=f"ssfr2 mismatch: transr={transr} uplo={uplo} n={n}",
    )


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_ssfr2_zero_alpha(n, uplo, transr):
    """alpha=0: arf must be unchanged."""
    rng = np.random.default_rng(seed=n * 7 + ord(uplo))
    A_np = _make_sym(n, rng)
    x_np = rng.standard_normal((n,)).astype(np.float32)
    y_np = rng.standard_normal((n,)).astype(np.float32)

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.strttf(A_t, transr=transr, uplo=uplo)
    arf0 = arf.clone()

    x_t = torch.from_numpy(x_np).cuda()
    y_t = torch.from_numpy(y_np).cuda()
    curfp.ssfr2(arf, x_t, y_t, alpha=0.0, transr=transr, uplo=uplo)

    assert torch.allclose(arf, arf0), (
        f"alpha=0: arf changed for transr={transr} uplo={uplo} n={n}"
    )


def test_ssfr2_raw_api():
    """ssfr2_raw produces the same result as the high-level API."""
    n = 8
    rng = np.random.default_rng(99)
    A_np = _make_sym(n, rng)
    x_np = rng.standard_normal((n,)).astype(np.float32)
    y_np = rng.standard_normal((n,)).astype(np.float32)
    alpha = 1.5

    A_t = torch.from_numpy(A_np).cuda()
    x_t = torch.from_numpy(x_np).cuda()
    y_t = torch.from_numpy(y_np).cuda()

    arf1 = curfp.strttf(A_t)
    arf2 = arf1.clone()

    curfp.ssfr2(arf1, x_t, y_t, alpha=alpha)
    curfp.ssfr2_raw(
        curfp.Handle(), curfp.OP_T, curfp.FILL_UPPER, n, alpha, x_t, 1, y_t, 1, arf2
    )

    assert torch.allclose(arf1, arf2, rtol=1e-5, atol=1e-5), (
        "ssfr2_raw result differs from high-level ssfr2"
    )


if __name__ == "__main__":
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32]:
        for transr in ["N", "T"]:
            for uplo in ["L", "U"]:
                test_ssfr2_vs_numpy(n, uplo, transr)
                test_ssfr2_zero_alpha(n, uplo, transr)
    test_ssfr2_raw_api()
    print("All ssfr2 tests PASSED!")
