"""
Tests for curfp.spftri — SPD matrix inversion from RFP Cholesky factor.

Validates all 8 RFP variants (transr N/T × uplo L/U × n even/odd) against
numpy.linalg.inv as reference.

Tests:
  - spftri output matches numpy.linalg.inv directly (spftrf → spftri)
  - spftri(spftrf(A)) @ A ≈ I  (residual test)
  - spftri followed by spftrs gives identity: solves X in A*X=B, X ≈ A^{-1}*B
  - Raw API smoke test
"""

import numpy as np
import pytest
import torch

import curfp

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TRANSR_VALS = ["N", "T"]
UPLO_VALS   = ["L", "U"]
N_VALS      = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128]

TOL_SMALL = 1e-4   # float32 tolerance for small n
TOL_LARGE = 1e-3   # float32 tolerance for larger n (condition number grows)


def make_spd_np(n: int, rng) -> np.ndarray:
    """Random SPD matrix: A = B @ B.T + n*I."""
    B = rng.standard_normal((n, n)).astype(np.float32)
    M = B @ B.T + n * np.eye(n, dtype=np.float32)
    return M


def tol_for_n(n: int) -> float:
    return TOL_LARGE if n >= 32 else TOL_SMALL


# ─────────────────────────────────────────────────────────────────────────────
# Core correctness: spftrf → spftri vs numpy.linalg.inv
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_spftri_vs_numpyinv(n, uplo, transr):
    """curfp.spftrf + curfp.spftri produces A^{-1} matching numpy.linalg.inv."""
    rng = np.random.default_rng(seed=17 + n)
    M_np = make_spd_np(n, rng)
    M_inv_ref = np.linalg.inv(M_np)

    # Prepare triangular input for spftrf
    if uplo == "U":
        tri_np = np.triu(M_np)
    else:
        tri_np = np.tril(M_np)

    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)

    # Factorize then invert
    info = curfp.spftrf(arf, transr=transr, uplo=uplo, check=False)
    assert info == 0, f"spftrf failed with info={info}"
    curfp.spftri(arf, transr=transr, uplo=uplo)

    # Unpack the RFP inverse to full matrix
    A_inv_gpu = curfp.stfttr(arf, transr=transr, uplo=uplo)
    A_inv_np = A_inv_gpu.cpu().numpy()

    # Compare only the stored triangle (other side is 0 from stfttr)
    if uplo == "U":
        mask = np.triu(np.ones((n, n), dtype=bool))
    else:
        mask = np.tril(np.ones((n, n), dtype=bool))

    tol = tol_for_n(n)
    np.testing.assert_allclose(
        A_inv_np[mask], M_inv_ref[mask], rtol=tol, atol=tol,
        err_msg=f"spftri mismatch: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Residual test: A_inv @ A ≈ I
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_spftri_residual(n, uplo, transr):
    """spftri: A^{-1} @ A ≈ I (Frobenius residual)."""
    rng = np.random.default_rng(seed=31 + n)
    M_np = make_spd_np(n, rng)

    if uplo == "U":
        tri_np = np.triu(M_np)
    else:
        tri_np = np.tril(M_np)

    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)

    curfp.spftrf(arf, transr=transr, uplo=uplo)
    curfp.spftri(arf, transr=transr, uplo=uplo)

    # Reconstruct full symmetric inverse
    A_inv = curfp.stfttr(arf, transr=transr, uplo=uplo).cpu().numpy()
    # Fill other triangle from symmetry
    if uplo == "U":
        A_inv = A_inv + A_inv.T - np.diag(np.diag(A_inv))
    else:
        A_inv = A_inv + A_inv.T - np.diag(np.diag(A_inv))

    if n == 1:
        # scalar case
        residual = abs(A_inv[0, 0] * M_np[0, 0] - 1.0)
    else:
        product = A_inv @ M_np
        residual = np.linalg.norm(product - np.eye(n)) / n

    tol = tol_for_n(n) * 10
    assert residual < tol, (
        f"spftri residual {residual:.2e} too large: "
        f"n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: spftrf + spftri + spftrs agreement
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [8, 16, 32, 64])
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_spftri_spftrs_consistency(n, uplo, transr):
    """A^{-1} @ b matches spftrs solution for the same system."""
    rng = np.random.default_rng(seed=53 + n)
    M_np = make_spd_np(n, rng)
    b_np = rng.standard_normal(n).astype(np.float32)

    if uplo == "U":
        tri_np = np.triu(M_np)
    else:
        tri_np = np.tril(M_np)

    # Solve via spftrs
    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf_solve = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)
    curfp.spftrf(arf_solve, transr=transr, uplo=uplo)
    B = torch.from_numpy(b_np.copy()).cuda().unsqueeze(1)
    curfp.spftrs(arf_solve, B, transr=transr, uplo=uplo)
    x_spftrs = B.squeeze(1).cpu().numpy()

    # Solve via spftri
    tri_gpu2 = torch.from_numpy(tri_np).cuda().contiguous()
    arf_inv = curfp.strttf(tri_gpu2, transr=transr, uplo=uplo)
    curfp.spftrf(arf_inv, transr=transr, uplo=uplo)
    curfp.spftri(arf_inv, transr=transr, uplo=uplo)

    A_inv = curfp.stfttr(arf_inv, transr=transr, uplo=uplo).cpu().numpy()
    if uplo == "U":
        A_inv = A_inv + A_inv.T - np.diag(np.diag(A_inv))
    else:
        A_inv = A_inv + A_inv.T - np.diag(np.diag(A_inv))
    x_inv = A_inv @ b_np

    np.testing.assert_allclose(
        x_spftrs, x_inv, rtol=1e-3, atol=1e-3,
        err_msg=f"spftri/spftrs mismatch: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raw API smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_spftri_raw_api():
    """spftri_raw produces the same result as the high-level API."""
    n = 8
    rng = np.random.default_rng(42)
    M_np = make_spd_np(n, rng)
    tri_np = np.triu(M_np)

    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()

    # High-level
    arf_hl = curfp.strttf(tri_gpu, transr="T", uplo="U")
    curfp.spftrf(arf_hl, transr="T", uplo="U")
    curfp.spftri(arf_hl, transr="T", uplo="U")

    # Raw
    arf_raw = curfp.strttf(tri_gpu, transr="T", uplo="U")
    with curfp.Handle() as h:
        info = curfp.spftrf_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, arf_raw)
        assert info == 0
        curfp.spftri_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, arf_raw)

    np.testing.assert_allclose(
        arf_raw.cpu().numpy(), arf_hl.cpu().numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="spftri_raw vs spftri mismatch"
    )
