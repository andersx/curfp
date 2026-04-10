"""
Tests for curfp.slansf — norm of a symmetric matrix in RFP format.

Validates all 4 transr/uplo combinations and three norm types against
numpy reference implementations for several matrix sizes.

Tests:
  - 1-norm (max absolute column sum) vs numpy.linalg.norm(A, 1)
  - Frobenius norm vs numpy.linalg.norm(A, 'fro')
  - Max-element norm vs numpy.max(|A|)
  - n=1 edge case
  - Raw API smoke test
"""

import numpy as np
import pytest
import torch

import curfp

TRANSR_VALS = ["N", "T"]
UPLO_VALS   = ["L", "U"]
N_VALS      = [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256]

# Relative tolerance for norm comparisons (float32 arithmetic)
RTOL = 1e-4


def make_spd(n: int, rng) -> np.ndarray:
    """Random n×n SPD matrix."""
    if n == 1:
        return np.array([[3.7]], dtype=np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    A = B @ B.T + np.eye(n, dtype=np.float32) * n
    return A


def to_rfp(A_np: np.ndarray, transr: str, uplo: str) -> torch.Tensor:
    """Convert a dense symmetric matrix to RFP format."""
    n = A_np.shape[0]
    if uplo == "U":
        tri_np = np.triu(A_np)
    else:
        tri_np = np.tril(A_np)
    tri_gpu = torch.from_numpy(tri_np).cuda().contiguous()
    return curfp.strttf(tri_gpu, transr=transr, uplo=uplo)


# ─────────────────────────────────────────────────────────────────────────────
# 1-norm
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_slansf_1norm(n, uplo, transr):
    """slansf('1') matches numpy 1-norm."""
    rng = np.random.default_rng(seed=42 + n)
    A_np = make_spd(n, rng)
    arf = to_rfp(A_np, transr, uplo)

    ref = float(np.linalg.norm(A_np, 1))
    got = curfp.slansf(arf, "1", transr=transr, uplo=uplo)

    assert abs(got - ref) <= RTOL * max(ref, 1e-10), (
        f"n={n}, transr={transr!r}, uplo={uplo!r}: got={got:.6e}, ref={ref:.6e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frobenius norm
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_slansf_frobenius(n, uplo, transr):
    """slansf('F') matches numpy Frobenius norm."""
    rng = np.random.default_rng(seed=100 + n)
    A_np = make_spd(n, rng)
    arf = to_rfp(A_np, transr, uplo)

    ref = float(np.linalg.norm(A_np, "fro"))
    got = curfp.slansf(arf, "F", transr=transr, uplo=uplo)

    assert abs(got - ref) <= RTOL * max(ref, 1e-10), (
        f"n={n}, transr={transr!r}, uplo={uplo!r}: got={got:.6e}, ref={ref:.6e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Max-element norm
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n",      N_VALS)
@pytest.mark.parametrize("uplo",   UPLO_VALS)
@pytest.mark.parametrize("transr", TRANSR_VALS)
def test_slansf_max(n, uplo, transr):
    """slansf('M') matches numpy max |element|."""
    rng = np.random.default_rng(seed=200 + n)
    A_np = make_spd(n, rng)
    arf = to_rfp(A_np, transr, uplo)

    ref = float(np.max(np.abs(A_np)))
    got = curfp.slansf(arf, "M", transr=transr, uplo=uplo)

    assert abs(got - ref) <= RTOL * max(ref, 1e-10), (
        f"n={n}, transr={transr!r}, uplo={uplo!r}: got={got:.6e}, ref={ref:.6e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_slansf_n0():
    """n=0: all norms return 0."""
    arf = torch.empty(0, dtype=torch.float32, device="cuda")
    for norm_str in ("1", "F", "M"):
        got = curfp.slansf(arf, norm_str, n=0)
        assert got == 0.0, f"n=0 norm={norm_str!r}: got {got}"


def test_slansf_n1():
    """n=1: A=[[v]], norms are all |v|."""
    v = 7.5
    arf = torch.tensor([v], dtype=torch.float32, device="cuda")
    # 1-norm = max col sum = v
    assert abs(curfp.slansf(arf, "1") - v) < 1e-5
    # Frobenius = sqrt(v²) = v
    assert abs(curfp.slansf(arf, "F") - v) < 1e-5
    # Max = v
    assert abs(curfp.slansf(arf, "M") - v) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Norm aliases
# ─────────────────────────────────────────────────────────────────────────────

def test_slansf_aliases():
    """Norm string aliases all produce the same result."""
    rng = np.random.default_rng(42)
    A_np = make_spd(16, rng)
    arf = to_rfp(A_np, "T", "U")

    one_ref = curfp.slansf(arf, "1")
    assert abs(curfp.slansf(arf, "O") - one_ref) < 1e-7
    assert abs(curfp.slansf(arf, "I") - one_ref) < 1e-7

    fro_ref = curfp.slansf(arf, "F")
    assert abs(curfp.slansf(arf, "E") - fro_ref) < 1e-7
    assert abs(curfp.slansf(arf, "fro") - fro_ref) < 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# Raw API smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_slansf_raw_api():
    """slansf_raw produces the same result as the high-level API."""
    rng = np.random.default_rng(7)
    A_np = make_spd(32, rng)
    arf = to_rfp(A_np, "T", "U")

    n = 32
    for norm_str, norm_int in [("1", curfp.NORM_ONE), ("F", curfp.NORM_FRO),
                                ("M", curfp.NORM_MAX)]:
        hl = curfp.slansf(arf, norm_str)
        with curfp.Handle() as h:
            raw = curfp.slansf_raw(h, norm_int, curfp.OP_T, curfp.FILL_UPPER, n, arf)
        assert abs(hl - raw) < 1e-7, (
            f"norm={norm_str!r}: slansf={hl}, slansf_raw={raw}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: slansf anorm → spftrf → spfcon round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_slansf_spfcon_roundtrip():
    """slansf('1') produces a valid anorm for spfcon."""
    import scipy.linalg
    import scipy.linalg.lapack

    rng = np.random.default_rng(77)
    n = 32
    A_np = make_spd(n, rng)
    arf = to_rfp(A_np, "T", "U")

    anorm_curfp = curfp.slansf(arf, "1")
    anorm_np    = float(np.linalg.norm(A_np, 1))
    assert abs(anorm_curfp - anorm_np) <= RTOL * anorm_np, (
        f"anorm mismatch: curfp={anorm_curfp:.6e}, np={anorm_np:.6e}"
    )

    curfp.spftrf(arf)
    rcond_curfp = curfp.spfcon(arf, anorm_curfp)

    # Reference via scipy
    c_upper = scipy.linalg.cholesky(A_np.astype(np.float64), lower=False)
    rcond_ref, info = scipy.linalg.lapack.spocon(c_upper, anorm_np)
    assert info == 0
    ratio = rcond_curfp / float(rcond_ref)
    assert 1/3 <= ratio <= 3, (
        f"rcond ratio out of range: {ratio:.3f} "
        f"(curfp={rcond_curfp:.4e}, ref={rcond_ref:.4e})"
    )
