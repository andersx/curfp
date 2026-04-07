"""
Python tests for curfp.spftrs — triangular solve using RFP Cholesky factor.

Each test:
1. Builds a random SPD matrix via ssfrk + diagonal regularization
2. Factorizes it with spftrf
3. Solves A * X = B with spftrs
4. Verifies residual ‖A_dense * X - B_orig‖ is small
"""

import math
import torch
import curfp


def _make_spd_rfp(n, k=None, transr="T", uplo="U", device="cuda"):
    """Return an RFP-packed SPD matrix and its dense equivalent."""
    if k is None:
        k = n
    A = torch.randn(n, k, dtype=torch.float32, device=device) / math.sqrt(k)
    C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device=device)
    curfp.ssfrk(A, C, transr=transr, uplo=uplo)
    curfp.add_to_diagonal(C, 1.0, transr=transr, uplo=uplo)

    # Dense version for residual check
    M_dense = A @ A.t()
    M_dense.diagonal().add_(1.0)
    return C, M_dense


def test_single_rhs_all_variants():
    """Single right-hand side, all 8 RFP variants (even and odd n)."""
    transr_vals = ["N", "T"]
    uplo_vals = ["L", "U"]
    n_vals = [4, 5]  # even, odd

    for transr in transr_vals:
        for uplo in uplo_vals:
            for n in n_vals:
                C, M_dense = _make_spd_rfp(n, k=n, transr=transr, uplo=uplo)

                B_orig = torch.randn(n, dtype=torch.float32, device="cuda")
                B = B_orig.clone()

                curfp.spftrf(C, transr=transr, uplo=uplo)
                curfp.spftrs(C, B, transr=transr, uplo=uplo)

                # Residual: ‖M_dense * X - B_orig‖
                residual = (M_dense @ B - B_orig).norm().item()
                assert residual < 1e-3, (
                    f"Residual {residual:.2e} too large for "
                    f"transr={transr} uplo={uplo} n={n}"
                )
    print("test_single_rhs_all_variants PASSED")


def test_multiple_rhs():
    """Multiple right-hand sides, default (optimal) variant."""
    n, k, nrhs = 64, 32, 10

    C, M_dense = _make_spd_rfp(n, k=k)
    B_orig = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    B = B_orig.clone()

    curfp.spftrf(C)
    curfp.spftrs(C, B)

    residual = (M_dense @ B - B_orig).norm().item()
    assert residual < 1e-2, f"Residual {residual:.2e} too large"
    print("test_multiple_rhs PASSED")


def test_large_n():
    """Larger matrix to exercise blocked code paths."""
    n, k, nrhs = 512, 256, 5

    C, M_dense = _make_spd_rfp(n, k=k)
    B_orig = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    B = B_orig.clone()

    curfp.spftrf(C)
    curfp.spftrs(C, B)

    # Relative residual
    rel_res = (M_dense @ B - B_orig).norm().item() / B_orig.norm().item()
    assert rel_res < 1e-3, f"Relative residual {rel_res:.2e} too large"
    print("test_large_n PASSED")


def test_high_level_api():
    """Verify the complete high-level pipeline: ssfrk → add_to_diagonal → spftrf → spftrs."""
    n, k, nrhs = 32, 16, 4

    A = torch.randn(n, k, dtype=torch.float32, device="cuda") / math.sqrt(k)
    C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
    B_orig = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    B = B_orig.clone()

    curfp.ssfrk(A, C)
    curfp.add_to_diagonal(C, 1.0)
    curfp.spftrf(C)
    curfp.spftrs(C, B)

    # Reconstruct dense M = A @ A.T + I and check residual
    M_dense = A @ A.t()
    M_dense.diagonal().add_(1.0)
    residual = (M_dense @ B - B_orig).norm().item()
    assert residual < 1e-2, f"Residual {residual:.2e} too large"
    print("test_high_level_api PASSED")


def test_spftrf_check_raises():
    """spftrf with check=True raises LinAlgError for non-SPD input."""
    # A single negative element is not positive definite
    C = torch.tensor([-1.0], dtype=torch.float32, device="cuda")
    raised = False
    try:
        curfp.spftrf(C, check=True)
    except (torch.linalg.LinAlgError, RuntimeError):
        raised = True
    assert raised, "Expected LinAlgError for non-SPD input"
    print("test_spftrf_check_raises PASSED")


def test_spftrf_check_false_returns_info():
    """spftrf with check=False returns info > 0 for non-SPD input."""
    C = torch.tensor([-1.0], dtype=torch.float32, device="cuda")
    info = curfp.spftrf(C, check=False)
    assert info > 0, f"Expected info > 0 for non-SPD, got {info}"
    print("test_spftrf_check_false_returns_info PASSED")


def test_rfp_diag_indices():
    """rfp_diag_indices returns correct indices for all 8 variants."""
    for n in [4, 5]:
        for transr in ["N", "T"]:
            for uplo in ["L", "U"]:
                idx = curfp.rfp_diag_indices(n, transr=transr, uplo=uplo, device="cuda")
                assert idx.shape == (n,), f"Expected {n} indices, got {idx.shape}"
                assert idx.unique().shape == (n,), "Indices are not unique"
                assert idx.max().item() < n * (n + 1) // 2, "Index out of range"
    print("test_rfp_diag_indices PASSED")


def test_add_to_diagonal():
    """add_to_diagonal correctly shifts all diagonal elements by the given value."""
    n = 6
    # Build identity-like matrix: ssfrk with I gives I @ I.T = I
    A = torch.eye(n, dtype=torch.float32, device="cuda")
    C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
    curfp.ssfrk(A, C)  # C = I in RFP

    # All diagonal elements should be 1.0
    idx = curfp.rfp_diag_indices(n)
    diag_before = C[idx].cpu()
    assert torch.allclose(diag_before, torch.ones(n)), (
        f"Expected all 1s, got {diag_before}"
    )

    # Shift by 2.5
    curfp.add_to_diagonal(C, 2.5)
    diag_after = C[idx].cpu()
    assert torch.allclose(diag_after, torch.full((n,), 3.5)), (
        f"Expected all 3.5, got {diag_after}"
    )
    print("test_add_to_diagonal PASSED")


def test_stream():
    """Verify spftrs works correctly on a non-default CUDA stream."""
    stream = torch.cuda.Stream()
    n, nrhs = 16, 3

    C, M_dense = _make_spd_rfp(n)
    B_orig = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
    B = B_orig.clone()

    curfp.spftrf(C)

    curfp.set_stream(stream)
    with torch.cuda.stream(stream):
        curfp.spftrs(C, B)
    torch.cuda.current_stream().wait_stream(stream)
    curfp.set_stream(None)

    residual = (M_dense @ B - B_orig).norm().item()
    assert residual < 1e-2, f"Residual {residual:.2e} too large on stream"
    print("test_stream PASSED")


if __name__ == "__main__":
    test_single_rhs_all_variants()
    test_multiple_rhs()
    test_large_n()
    test_high_level_api()
    test_spftrf_check_raises()
    test_spftrf_check_false_returns_info()
    test_rfp_diag_indices()
    test_add_to_diagonal()
    test_stream()
    print("\nAll spftrs tests PASSED!")
