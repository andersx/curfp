"""
Tests for curfp.strttf and curfp.stfttr — RFP ↔ full triangular conversion.

Validates all 8 RFP variants (transr N/T × uplo L/U × n even/odd) against
scipy.linalg.lapack.strttf and scipy.linalg.lapack.stfttr as reference.

Round-trip tests: strttf(stfttr(arf)) == arf  and  stfttr(strttf(A)) ⊇ A (triangle).
"""

import math
import numpy as np
import pytest
import torch
from scipy.linalg import lapack as spla

import curfp

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TRANSR_VALS = ["N", "T"]
UPLO_VALS   = ["L", "U"]
# n values: small odd, small even, larger odd, larger even
N_VALS      = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 63, 64, 65, 128, 129, 257]

TOL = 1e-6


def make_lower_tri_np(n: int, rng) -> np.ndarray:
    """Random lower triangular n×n matrix (row-major NumPy)."""
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1):
            A[i, j] = rng.standard_normal()
    return A


def make_upper_tri_np(n: int, rng) -> np.ndarray:
    """Random upper triangular n×n matrix (row-major NumPy)."""
    return make_lower_tri_np(n, rng).T.copy()


def np_to_rfp(A_np: np.ndarray, transr: str, uplo: str) -> np.ndarray:
    """Reference: full triangular → RFP using scipy."""
    n = A_np.shape[0]
    # scipy strttf expects A in column-major (Fortran order).
    # Our A_np is row-major; the same mathematical elements are at A_np[i,j].
    # scipy uses col-major internally but accesses A_np[i,j] correctly since
    # we pass the array (scipy handles transposition transparently).
    arf, info = spla.strttf(A_np, transr=transr.encode(), uplo=uplo.encode())
    assert info == 0, f"scipy strttf returned info={info}"
    return arf.astype(np.float32)


def rfp_to_np(arf_np: np.ndarray, n: int, transr: str, uplo: str) -> np.ndarray:
    """Reference: RFP → full triangular using scipy."""
    A, info = spla.stfttr(n, arf_np, transr=transr.encode(), uplo=uplo.encode())
    assert info == 0, f"scipy stfttr returned info={info}"
    return A.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# strttf: full → RFP
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_strttf_vs_scipy(n, uplo, transr):
    """curfp.strttf output matches scipy.linalg.lapack.strttf."""
    rng = np.random.default_rng(seed=42 + n)

    A_np = make_upper_tri_np(n, rng) if uplo == "U" else make_lower_tri_np(n, rng)
    arf_ref = np_to_rfp(A_np, transr, uplo)

    A_gpu  = torch.from_numpy(A_np).cuda().contiguous()
    arf_gpu = curfp.strttf(A_gpu, transr=transr, uplo=uplo)

    arf_gpu_np = arf_gpu.cpu().numpy()
    np.testing.assert_allclose(
        arf_gpu_np, arf_ref, rtol=0, atol=TOL,
        err_msg=f"strttf mismatch: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# stfttr: RFP → full
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_stfttr_vs_scipy(n, uplo, transr):
    """curfp.stfttr output matches scipy.linalg.lapack.stfttr."""
    rng = np.random.default_rng(seed=99 + n)

    nt = n * (n + 1) // 2
    arf_np = rng.standard_normal(nt).astype(np.float32)
    A_ref  = rfp_to_np(arf_np, n, transr, uplo)

    arf_gpu = torch.from_numpy(arf_np).cuda()
    A_gpu   = curfp.stfttr(arf_gpu, transr=transr, uplo=uplo)

    A_gpu_np = A_gpu.cpu().numpy()

    # Extract the triangle that was written (other triangle is 0 on GPU).
    if uplo == "U":
        mask = np.triu(np.ones((n, n), dtype=bool))
    else:
        mask = np.tril(np.ones((n, n), dtype=bool))

    np.testing.assert_allclose(
        A_gpu_np[mask], A_ref[mask], rtol=0, atol=TOL,
        err_msg=f"stfttr mismatch: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_round_trip_rfp_to_full_to_rfp(n, uplo, transr):
    """strttf(stfttr(arf)) == arf."""
    rng = np.random.default_rng(seed=7 + n)
    nt = n * (n + 1) // 2
    arf = torch.from_numpy(rng.standard_normal(nt).astype(np.float32)).cuda()

    A    = curfp.stfttr(arf, transr=transr, uplo=uplo)
    arf2 = curfp.strttf(A,   transr=transr, uplo=uplo)

    np.testing.assert_allclose(
        arf2.cpu().numpy(), arf.cpu().numpy(), rtol=0, atol=TOL,
        err_msg=f"round-trip (rfp→full→rfp) failed: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_round_trip_full_to_rfp_to_full(n, uplo, transr):
    """stfttr(strttf(A)) recovers A on the specified triangle."""
    rng = np.random.default_rng(seed=13 + n)
    A_np = make_upper_tri_np(n, rng) if uplo == "U" else make_lower_tri_np(n, rng)

    A    = torch.from_numpy(A_np).cuda().contiguous()
    arf  = curfp.strttf(A, transr=transr, uplo=uplo)
    A2   = curfp.stfttr(arf, transr=transr, uplo=uplo)

    if uplo == "U":
        mask = np.triu(np.ones((n, n), dtype=bool))
    else:
        mask = np.tril(np.ones((n, n), dtype=bool))

    A2_np = A2.cpu().numpy()
    np.testing.assert_allclose(
        A2_np[mask], A_np[mask], rtol=0, atol=TOL,
        err_msg=f"round-trip (full→rfp→full) failed: n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: strttf → spftrf → spftrs pipeline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_strttf_in_pipeline(n, uplo, transr):
    """strttf produces an RFP tensor that spftrf/spftrs can use correctly."""
    rng = np.random.default_rng(seed=31 + n)
    k   = n

    # Build an SPD matrix via A @ A.T + I
    A_np = rng.standard_normal((n, k)).astype(np.float32) / math.sqrt(k)
    M_np = A_np @ A_np.T + np.eye(n, dtype=np.float32)

    # Dense → triangular → RFP using strttf
    if uplo == "U":
        tri_np = np.triu(M_np)
    else:
        tri_np = np.tril(M_np)

    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    arf     = curfp.strttf(tri_gpu, transr=transr, uplo=uplo)

    # Solve via RFP Cholesky
    curfp.spftrf(arf, transr=transr, uplo=uplo)
    b_np  = rng.standard_normal((n, 3)).astype(np.float32)
    B     = torch.from_numpy(b_np).cuda().contiguous()
    curfp.spftrs(arf, B, transr=transr, uplo=uplo)

    # Verify: M_np @ X ≈ b
    X_np = B.cpu().numpy()
    residual = np.linalg.norm(M_np @ X_np - b_np) / (np.linalg.norm(b_np) + 1e-12)
    assert residual < 1e-3, (
        f"pipeline residual {residual:.2e} too large: "
        f"n={n}, transr={transr!r}, uplo={uplo!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raw API smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_raw_api():
    """strttf_raw and stfttr_raw produce the same result as high-level API."""
    n = 8
    rng = np.random.default_rng(42)
    A_np = make_upper_tri_np(n, rng)
    A    = torch.from_numpy(A_np).cuda().contiguous()

    with curfp.Handle() as h:
        # strttf_raw
        arf_raw = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
        curfp.strttf_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, A, n, arf_raw)

        # stfttr_raw
        A2_raw = torch.zeros(n, n, dtype=torch.float32, device="cuda")
        curfp.stfttr_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, arf_raw, A2_raw, n)

    # Compare against high-level
    arf_hl = curfp.strttf(A, transr="T", uplo="U")
    A2_hl  = curfp.stfttr(arf_hl, transr="T", uplo="U")

    np.testing.assert_allclose(arf_raw.cpu().numpy(), arf_hl.cpu().numpy(), atol=TOL)
    np.testing.assert_allclose(A2_raw.cpu().numpy(), A2_hl.cpu().numpy(), atol=TOL)
