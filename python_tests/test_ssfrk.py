"""
Python tests for curfp.ssfrk.

RFP layout for n=2, TRANSR=N, UPLO=L (Case 5):
  ARF[0] = C(1,1),  ARF[1] = C(0,0),  ARF[2] = C(1,0)
"""

import torch
import curfp


def test_passthrough_beta1():
    """alpha=0, beta=1 → C must be unchanged for all 8 RFP variants."""
    transr_vals = [curfp.OP_N, curfp.OP_T]
    uplo_vals   = [curfp.FILL_LOWER, curfp.FILL_UPPER]
    n_vals      = [5, 6]

    with curfp.Handle() as h:
        for transr in transr_vals:
            for uplo in uplo_vals:
                for n in n_vals:
                    nt = n * (n + 1) // 2
                    C = torch.randn(nt, dtype=torch.float32, device="cuda")
                    C_orig = C.clone()
                    A_dummy = torch.zeros(1, dtype=torch.float32, device="cuda")

                    curfp.ssfrk(h, transr, uplo, curfp.OP_N,
                                n, 1, 0.0, A_dummy, n, 1.0, C)

                    assert torch.allclose(C, C_orig), (
                        f"passthrough failed for transr={transr} uplo={uplo} n={n}"
                    )
    print("test_passthrough_beta1 PASSED")


def test_zero_beta0():
    """alpha=0, beta=0 → C must become all zeros."""
    transr_vals = [curfp.OP_N, curfp.OP_T]
    uplo_vals   = [curfp.FILL_LOWER, curfp.FILL_UPPER]
    n_vals      = [5, 6]

    with curfp.Handle() as h:
        for transr in transr_vals:
            for uplo in uplo_vals:
                for n in n_vals:
                    nt = n * (n + 1) // 2
                    C = torch.full((nt,), 99.0, dtype=torch.float32, device="cuda")
                    A_dummy = torch.zeros(1, dtype=torch.float32, device="cuda")

                    curfp.ssfrk(h, transr, uplo, curfp.OP_N,
                                n, 1, 0.0, A_dummy, n, 0.0, C)

                    assert C.abs().max().item() < 1e-5, (
                        f"zero failed for transr={transr} uplo={uplo} n={n}"
                    )
    print("test_zero_beta0 PASSED")


def test_n1_rank1():
    """n=1: C := alpha*a^2 + beta*c for all 4 transr/uplo combinations."""
    a, c = 3.0, 5.0
    alpha, beta = 2.0, 0.5
    expected = alpha * a * a + beta * c

    transr_vals = [curfp.OP_N, curfp.OP_T]
    uplo_vals   = [curfp.FILL_LOWER, curfp.FILL_UPPER]

    with curfp.Handle() as h:
        for transr in transr_vals:
            for uplo in uplo_vals:
                C = torch.tensor([c], dtype=torch.float32, device="cuda")
                A = torch.tensor([a], dtype=torch.float32, device="cuda")

                curfp.ssfrk(h, transr, uplo, curfp.OP_N,
                            1, 1, alpha, A, 1, beta, C)

                result = C.cpu().item()
                assert abs(result - expected) < 1e-4, (
                    f"n=1 rank-1 failed: got {result}, expected {expected}"
                )
    print("test_n1_rank1 PASSED")


def test_n2_NL_rank1():
    """
    n=2, TRANSR=N, UPLO=L, TRANS=N, k=1 — numerical correctness.

    C := alpha * [a0, a1]^T * [a0, a1] + beta * C_init
    """
    a0, a1 = 2.0, 3.0
    c00, c10, c11 = 1.0, 0.5, 4.0
    alpha, beta = 2.0, 0.5

    # RFP encoding: ARF[0]=C(1,1), ARF[1]=C(0,0), ARF[2]=C(1,0)
    C = torch.tensor([c11, c00, c10], dtype=torch.float32, device="cuda")
    A = torch.tensor([a0, a1],        dtype=torch.float32, device="cuda")  # lda=2

    with curfp.Handle() as h:
        curfp.ssfrk(h, curfp.OP_N, curfp.FILL_LOWER, curfp.OP_N,
                    2, 1, alpha, A, 2, beta, C)

    ref00 = alpha * a0 * a0 + beta * c00
    ref10 = alpha * a1 * a0 + beta * c10
    ref11 = alpha * a1 * a1 + beta * c11

    result = C.cpu().tolist()
    tol = 1e-4
    assert abs(result[1] - ref00) < tol, f"C(0,0): got {result[1]}, expected {ref00}"
    assert abs(result[2] - ref10) < tol, f"C(1,0): got {result[2]}, expected {ref10}"
    assert abs(result[0] - ref11) < tol, f"C(1,1): got {result[0]}, expected {ref11}"
    print("test_n2_NL_rank1 PASSED")


def test_stream():
    """Verify set_stream does not crash and computation still works."""
    stream = torch.cuda.Stream()

    a, c = 2.0, 0.0
    alpha, beta = 1.0, 0.0
    C = torch.tensor([c], dtype=torch.float32, device="cuda")
    A = torch.tensor([a], dtype=torch.float32, device="cuda")

    with curfp.Handle() as h:
        h.set_stream(stream)
        with torch.cuda.stream(stream):
            curfp.ssfrk(h, curfp.OP_N, curfp.FILL_LOWER, curfp.OP_N,
                        1, 1, alpha, A, 1, beta, C)
        torch.cuda.current_stream().wait_stream(stream)

    expected = alpha * a * a
    assert abs(C.cpu().item() - expected) < 1e-5
    print("test_stream PASSED")


if __name__ == "__main__":
    test_passthrough_beta1()
    test_zero_beta0()
    test_n1_rank1()
    test_n2_NL_rank1()
    test_stream()
    print("\nAll ssfrk tests PASSED!")
