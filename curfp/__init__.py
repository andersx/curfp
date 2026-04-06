"""
curfp — CUDA RFP format matrix operations for PyTorch.

Usage::

    import torch
    import curfp

    with curfp.Handle() as h:
        # Symmetric rank-k update: C := alpha * A * A^T + beta * C
        curfp.ssfrk(h, curfp.OP_N, curfp.FILL_LOWER, curfp.OP_N,
                    n, k, alpha, A, lda, beta, C)

        # In-place Cholesky factorization (returns info, 0 = success)
        info = curfp.spftrf(h, curfp.OP_N, curfp.FILL_LOWER, n, A)

    # With an explicit CUDA stream
    stream = torch.cuda.Stream()
    with curfp.Handle() as h:
        h.set_stream(stream)
        ...

Constants:
    OP_N, OP_T              operation on A (no-transpose / transpose)
    FILL_LOWER, FILL_UPPER  which triangle of the symmetric matrix is stored
"""

from ._curfp_C import (
    Handle as _HandleBase,
    OP_N,
    OP_T,
    FILL_LOWER,
    FILL_UPPER,
    ssfrk  as _ssfrk_C,
    spftrf as _spftrf_C,
)

import torch


def _validate(t, name: str) -> None:
    """Check that t is a contiguous float32 CUDA tensor."""
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be float32 (got {t.dtype})")


class Handle(_HandleBase):
    """
    curfp library handle — wraps cuBLAS and cuSOLVER handles.
    Use as a context manager::

        with curfp.Handle() as h:
            curfp.ssfrk(h, ...)
    """

    def set_stream(self, stream: torch.cuda.Stream) -> None:
        """Bind this handle to a ``torch.cuda.Stream``."""
        self.set_stream_ptr(stream.cuda_stream)


def ssfrk(
    handle: Handle,
    transr: int,
    uplo: int,
    trans: int,
    n: int,
    k: int,
    alpha: float,
    A: torch.Tensor,
    lda: int,
    beta: float,
    C: torch.Tensor,
) -> None:
    """
    Symmetric rank-k update in RFP format (single precision, in-place on C).

    Computes::

        trans=OP_N:  C := alpha * A * A^T + beta * C   (A is n × k)
        trans=OP_T:  C := alpha * A^T * A + beta * C   (A is k × n)

    Args:
        handle: curfp.Handle
        transr: RFP storage variant — OP_N or OP_T
        uplo:   stored triangle — FILL_LOWER or FILL_UPPER
        trans:  operation on A — OP_N or OP_T
        n:      order of symmetric matrix C
        k:      columns (trans=OP_N) or rows (trans=OP_T) of A
        alpha:  scalar multiplier for A * A^T
        A:      float32 CUDA tensor, shape (n, k) for OP_N or (k, n) for OP_T
        lda:    leading dimension of A
        beta:   scalar multiplier for C
        C:      float32 CUDA tensor, RFP format, n*(n+1)//2 elements
    """
    _validate(A, "A")
    _validate(C, "C")
    _ssfrk_C(handle, transr, uplo, trans, n, k,
              float(alpha), A.data_ptr(), lda,
              float(beta),  C.data_ptr())


def spftrf(
    handle: Handle,
    transr: int,
    uplo: int,
    n: int,
    A: torch.Tensor,
) -> int:
    """
    Cholesky factorization in RFP format (single precision, in-place on A).

    Computes::

        uplo=FILL_LOWER:  A = L * L^T
        uplo=FILL_UPPER:  A = U^T * U

    Args:
        handle: curfp.Handle
        transr: RFP storage variant — OP_N or OP_T
        uplo:   stored triangle — FILL_LOWER or FILL_UPPER
        n:      order of matrix A
        A:      float32 CUDA tensor, RFP format, n*(n+1)//2 elements

    Returns:
        info (int): 0 = success; >0 = leading minor of order info is not
        positive definite.
    """
    _validate(A, "A")
    return _spftrf_C(handle, transr, uplo, n, A.data_ptr())


__all__ = [
    "Handle",
    "ssfrk",
    "spftrf",
    "OP_N",
    "OP_T",
    "FILL_LOWER",
    "FILL_UPPER",
]
