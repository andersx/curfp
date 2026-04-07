"""
Python tests for curfp.spftrf.

RFP layout references (derived from pointer offsets in curfp_spftrf.cpp):

  n=2, TRANSR=N, UPLO=L (Case 5, k=1, lda_rfp=3):
    ARF[0]=A(1,1),  ARF[1]=A(0,0),  ARF[2]=A(1,0)

  n=3, TRANSR=N, UPLO=L (Case 1, n1=2, n2=1, lda_rfp=3):
    ARF[0]=A(0,0),  ARF[1]=A(1,0),  ARF[2]=A(2,0)
    ARF[3]=A(2,2),  ARF[4]=A(1,1),  ARF[5]=A(2,1)
"""

import math
import torch
import curfp


def test_n1_all_variants():
    """n=1: all 4 transr/uplo variants → single element becomes sqrt(val)."""
    val = 9.0
    expected = math.sqrt(val)

    transr_vals = [curfp.OP_N, curfp.OP_T]
    uplo_vals   = [curfp.FILL_LOWER, curfp.FILL_UPPER]

    with curfp.Handle() as h:
        for transr in transr_vals:
            for uplo in uplo_vals:
                A = torch.tensor([val], dtype=torch.float32, device="cuda")
                info = curfp.spftrf_raw(h, transr, uplo, 1, A)

                assert info == 0, f"info={info} for transr={transr} uplo={uplo}"
                result = A.cpu().item()
                assert abs(result - expected) < 1e-5, (
                    f"n=1: got {result}, expected {expected} "
                    f"(transr={transr} uplo={uplo})"
                )
    print("test_n1_all_variants PASSED")


def test_n2_NL():
    """
    n=2, TRANSR=N, UPLO=L (Case 5) — numerical Cholesky verification.

    Input A = [[4, 2], [2, 3]]
    RFP encoding: ARF = {A(1,1), A(0,0), A(1,0)} = {3, 4, 2}

    Expected L:
      L(0,0) = sqrt(4)   = 2
      L(1,0) = 2/2       = 1
      L(1,1) = sqrt(3-1) = sqrt(2)

    Output RFP = {L(1,1), L(0,0), L(1,0)} = {sqrt(2), 2, 1}
    """
    a00, a10, a11 = 4.0, 2.0, 3.0
    A = torch.tensor([a11, a00, a10], dtype=torch.float32, device="cuda")

    exp_l00 = math.sqrt(a00)
    exp_l10 = a10 / exp_l00
    exp_l11 = math.sqrt(a11 - exp_l10 ** 2)

    with curfp.Handle() as h:
        info = curfp.spftrf_raw(h, curfp.OP_N, curfp.FILL_LOWER, 2, A)

    assert info == 0, f"info={info}"

    result = A.cpu().tolist()
    tol = 1e-4
    # Output encoding: result[1]=L(0,0), result[2]=L(1,0), result[0]=L(1,1)
    assert abs(result[1] - exp_l00) < tol, f"L(0,0): got {result[1]}, expected {exp_l00}"
    assert abs(result[2] - exp_l10) < tol, f"L(1,0): got {result[2]}, expected {exp_l10}"
    assert abs(result[0] - exp_l11) < tol, f"L(1,1): got {result[0]}, expected {exp_l11}"
    print("test_n2_NL PASSED")


def test_n3_NL():
    """
    n=3, TRANSR=N, UPLO=L (Case 1) — full Cholesky factor verification.

    A = [[5, 1, 2], [1, 4, 0], [2, 0, 6]]
    RFP = { A(0,0), A(1,0), A(2,0), A(2,2), A(1,1), A(2,1) }
         = {  5,      1,      2,      6,      4,      0      }

    Cholesky L (expected):
      L(0,0) = sqrt(5)
      L(1,0) = 1/sqrt(5)
      L(2,0) = 2/sqrt(5)
      L(1,1) = sqrt(4 - 1/5) = sqrt(19/5)
      L(2,1) = (0 - (2/sqrt(5))*(1/sqrt(5))) / sqrt(19/5)
      L(2,2) = sqrt(6 - L(2,0)^2 - L(2,1)^2)

    Output RFP = { L(0,0), L(1,0), L(2,0), L(2,2), L(1,1), L(2,1) }
    """
    a00, a10, a20 = 5.0, 1.0, 2.0
    a11, a21      = 4.0, 0.0
    a22           = 6.0

    A = torch.tensor([a00, a10, a20, a22, a11, a21],
                     dtype=torch.float32, device="cuda")

    # Reference Cholesky
    exp_l00 = math.sqrt(a00)
    exp_l10 = a10 / exp_l00
    exp_l20 = a20 / exp_l00
    exp_l11 = math.sqrt(a11 - exp_l10 ** 2)
    exp_l21 = (a21 - exp_l20 * exp_l10) / exp_l11
    exp_l22 = math.sqrt(a22 - exp_l20 ** 2 - exp_l21 ** 2)

    with curfp.Handle() as h:
        info = curfp.spftrf_raw(h, curfp.OP_N, curfp.FILL_LOWER, 3, A)

    assert info == 0, f"info={info}"

    # Output RFP has the same layout as input
    result = A.cpu().tolist()
    tol = 1e-4
    checks = [
        (result[0], exp_l00, "L(0,0)"),
        (result[1], exp_l10, "L(1,0)"),
        (result[2], exp_l20, "L(2,0)"),
        (result[4], exp_l11, "L(1,1)"),
        (result[5], exp_l21, "L(2,1)"),
        (result[3], exp_l22, "L(2,2)"),
    ]
    for got, exp, name in checks:
        assert abs(got - exp) < tol, f"{name}: got {got:.6f}, expected {exp:.6f}"
    print("test_n3_NL PASSED")


def test_not_positive_definite():
    """A non-positive-definite matrix should return info > 0."""
    # A = [[-1]] — not positive definite
    A = torch.tensor([-1.0], dtype=torch.float32, device="cuda")
    with curfp.Handle() as h:
        info = curfp.spftrf_raw(h, curfp.OP_N, curfp.FILL_LOWER, 1, A)
    assert info > 0, f"Expected info > 0 for non-SPD matrix, got {info}"
    print("test_not_positive_definite PASSED")


def test_stream():
    """Verify set_stream does not crash and factorization still works."""
    stream = torch.cuda.Stream()
    val = 4.0
    A = torch.tensor([val], dtype=torch.float32, device="cuda")

    with curfp.Handle() as h:
        h.set_stream(stream)
        with torch.cuda.stream(stream):
            info = curfp.spftrf_raw(h, curfp.OP_N, curfp.FILL_LOWER, 1, A)
        torch.cuda.current_stream().wait_stream(stream)

    assert info == 0
    assert abs(A.cpu().item() - math.sqrt(val)) < 1e-5
    print("test_stream PASSED")


if __name__ == "__main__":
    test_n1_all_variants()
    test_n2_NL()
    test_n3_NL()
    test_not_positive_definite()
    test_stream()
    print("\nAll spftrf tests PASSED!")
