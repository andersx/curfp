"""Tests for curfp.dstrttf / curfp.dstfttr — RFP ↔ full triangular conversion (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128]
TOL = 1e-13


def _make_sym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_round_trip(n, uplo, transr):
    """dstrttf followed by dstfttr recovers the original triangle (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr))
    A_np = _make_sym(n, rng)

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    A_out = curfp.dstfttr(arf, transr=transr, uplo=uplo)
    A_out_np = A_out.cpu().numpy()

    if uplo == "L":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        mask = np.triu(np.ones((n, n), dtype=bool))

    np.testing.assert_allclose(
        A_out_np[mask],
        A_np[mask],
        rtol=TOL,
        atol=TOL,
        err_msg=f"round-trip mismatch: transr={transr} uplo={uplo} n={n}",
    )


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_nt_size(n, uplo, transr):
    """RFP array has n*(n+1)//2 elements (float64)."""
    rng = np.random.default_rng(seed=n * 7)
    A_np = _make_sym(n, rng)
    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    expected_size = n * (n + 1) // 2
    assert arf.numel() == expected_size, (
        f"arf has {arf.numel()} elements, expected {expected_size} "
        f"(transr={transr} uplo={uplo} n={n})"
    )
    assert arf.dtype == torch.float64


def test_dstrttf_raw_api():
    """dstrttf_raw / dstfttr_raw produce same result as high-level API."""
    n = 8
    rng = np.random.default_rng(42)
    A_np = _make_sym(n, rng)

    A_t = torch.from_numpy(A_np).cuda()
    nt = n * (n + 1) // 2

    arf_hi = curfp.dstrttf(A_t)

    arf_lo = torch.empty(nt, dtype=torch.float64, device="cuda")
    curfp.dstrttf_raw(curfp.Handle(), curfp.OP_T, curfp.FILL_UPPER, n, A_t, n, arf_lo)

    assert torch.allclose(arf_hi, arf_lo, rtol=1e-14, atol=1e-14)
