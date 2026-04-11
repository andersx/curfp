"""Tests for curfp.dlansf — matrix norm in RFP format (double precision)."""

import numpy as np
import pytest
import torch
import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS = ["L", "U"]
N_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64]
NORM_VALS = ["M", "1", "F"]
TOL = 1e-10


def _make_sym(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2.0


@pytest.mark.parametrize("n", N_VALS)
@pytest.mark.parametrize("uplo", UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
@pytest.mark.parametrize("norm", NORM_VALS)
def test_dlansf_vs_numpy(norm, transr, uplo, n):
    """curfp.dlansf norm matches numpy (float64)."""
    rng = np.random.default_rng(seed=n * 100 + ord(uplo) + ord(transr) + ord(norm))
    A_np = _make_sym(n, rng)

    # numpy reference — A_np is already symmetric (from _make_sym)
    A_full = A_np
    if norm == "M":
        ref = np.max(np.abs(A_full))
    elif norm == "1":
        ref = np.max(np.sum(np.abs(A_full), axis=0))
    else:  # F
        ref = np.linalg.norm(A_full, "fro")

    A_t = torch.from_numpy(A_np).cuda()
    arf = curfp.dstrttf(A_t, transr=transr, uplo=uplo)
    result = curfp.dlansf(arf, norm=norm, transr=transr, uplo=uplo)

    assert abs(result - ref) / max(abs(ref), 1e-15) < 1e-9, (
        f"dlansf({norm}) mismatch: got {result}, expected {ref} "
        f"(transr={transr} uplo={uplo} n={n})"
    )
