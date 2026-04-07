"""
curfp — CUDA RFP format matrix operations for PyTorch.

High-level API (preferred)::

    import torch
    import curfp

    n, k, nrhs = 4096, 128, 10
    A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
    C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
    B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")

    curfp.ssfrk(A, C)                        # C = A @ A.T in RFP format
    C[curfp.rfp_diag_indices(n)] += 1.0      # regularize diagonal
    curfp.spftrf(C)                          # Cholesky factorization in-place
    curfp.spftrs(C, B)                       # solve (A@A.T + I) X = B in-place

    # Equivalently, add_to_diagonal handles the regularization:
    curfp.ssfrk(A, C)
    curfp.add_to_diagonal(C, 1.0)
    curfp.spftrf(C)
    curfp.spftrs(C, B)

Low-level API (explicit handle, integer enums, all parameters)::

    with curfp.Handle() as h:
        curfp.ssfrk_raw(h, curfp.OP_T, curfp.FILL_UPPER, curfp.OP_T,
                        n, k, 1.0, A, k, 0.0, C)
        info = curfp.spftrf_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, C)
        curfp.spftrs_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, nrhs, C, B, n)

Stream control::

    stream = torch.cuda.Stream()
    curfp.set_stream(stream)
    curfp.ssfrk(A, C)
    ...
    curfp.set_stream(None)   # revert to default stream
"""

from ._curfp_C import (
    Handle as _HandleBase,
    OP_N,
    OP_T,
    FILL_LOWER,
    FILL_UPPER,
    ssfrk as _ssfrk_C,
    spftrf as _spftrf_C,
    spftrs as _spftrs_C,
)

import math
import torch

# ---------------------------------------------------------------------------
# Global handle cache (one handle per CUDA device, lazily created)
# ---------------------------------------------------------------------------
_handles: dict = {}


def _get_handle() -> "_HandleBase":
    dev = torch.cuda.current_device()
    if dev not in _handles:
        _handles[dev] = Handle()
    return _handles[dev]


def set_stream(stream) -> None:
    """Bind all curfp operations on the current device to ``stream``.

    Pass ``None`` or a stream with ``cuda_stream == 0`` to revert to the
    default (null) CUDA stream.
    """
    h = _get_handle()
    ptr = 0 if stream is None else stream.cuda_stream
    h.set_stream_ptr(ptr)


# ---------------------------------------------------------------------------
# String → enum helpers
# ---------------------------------------------------------------------------
_OP_MAP = {"N": OP_N, "T": OP_T}
_FILL_MAP = {"L": FILL_LOWER, "U": FILL_UPPER}


def _op(s: str) -> int:
    try:
        return _OP_MAP[s.upper()]
    except KeyError:
        raise ValueError(f"transr/trans must be 'N' or 'T', got {s!r}")


def _fill(s: str) -> int:
    try:
        return _FILL_MAP[s.upper()]
    except KeyError:
        raise ValueError(f"uplo must be 'L' or 'U', got {s!r}")


# ---------------------------------------------------------------------------
# Dimension inference helpers
# ---------------------------------------------------------------------------
def _n_from_rfp(numel: int) -> int:
    """Recover n from n*(n+1)//2 == numel."""
    n = int((-1.0 + math.sqrt(1.0 + 8.0 * numel)) / 2.0)
    if n * (n + 1) // 2 != numel:
        raise ValueError(
            f"Tensor has {numel} elements which is not a valid RFP size n*(n+1)//2"
        )
    return n


# ---------------------------------------------------------------------------
# Tensor validation
# ---------------------------------------------------------------------------
def _validate(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be float32 (got {t.dtype})")


# ---------------------------------------------------------------------------
# RFP diagonal utilities
# ---------------------------------------------------------------------------
def rfp_diag_indices(
    n: int, transr: str = "T", uplo: str = "U", device=None
) -> torch.Tensor:
    """Return the flat RFP-array indices of the n diagonal elements.

    Covers all 8 RFP storage variants (even/odd n × transr N/T × uplo L/U).
    Derived from the sub-block pointer offsets in curfp_ssfrk.cpp.

    Args:
        n:      Matrix order.
        transr: RFP storage variant, 'N' or 'T' (default 'T').
        uplo:   Triangle stored, 'L' or 'U' (default 'U').
        device: Target CUDA device (default: current device).

    Returns:
        1-D int64 tensor of length n with flat indices into the RFP array.
    """
    if device is None:
        device = torch.cuda.current_device()

    transr_n = transr.upper() == "N"
    lower = uplo.upper() == "L"

    if n % 2 != 0:
        # Odd n
        if lower:
            n1, n2 = n - n // 2, n // 2  # n1 = ceil(n/2)
        else:
            n1, n2 = n // 2, n - n // 2

        if transr_n:
            # Cases 1 (L) and 2 (U): lda_rfp = n
            if lower:
                # L11 at C+0 (n1×n1), L22 at C+n (n2×n2)
                d1 = torch.arange(n1, device=device) * (n + 1)
                d2 = n + torch.arange(n2, device=device) * (n + 1)
            else:
                # U11 at C+n2 (n1×n1), U22 at C+n1 (n2×n2)
                d1 = n2 + torch.arange(n1, device=device) * (n + 1)
                d2 = n1 + torch.arange(n2, device=device) * (n + 1)
        else:
            # Cases 3 (L) and 4 (U): transr=T
            if lower:
                # lda_rfp = n1; L11 at C+0 (upper), L22 at C+1 (lower)
                d1 = torch.arange(n1, device=device) * (n1 + 1)
                d2 = 1 + torch.arange(n2, device=device) * (n1 + 1)
            else:
                # lda_rfp = n2; U11 at C+n2*n2 (upper), U22 at C+n1*n2 (lower)
                d1 = n2 * n2 + torch.arange(n1, device=device) * (n2 + 1)
                d2 = n1 * n2 + torch.arange(n2, device=device) * (n2 + 1)
    else:
        # Even n
        nk = n // 2
        if transr_n:
            # Cases 5 (L) and 6 (U): lda_rfp = n+1
            step = n + 2  # lda_rfp + 1
            if lower:
                # L11 at C+1, L22 at C+0
                d1 = 1 + torch.arange(nk, device=device) * step
                d2 = torch.arange(nk, device=device) * step
            else:
                # U11 at C+nk+1, U22 at C+nk
                d1 = (nk + 1) + torch.arange(nk, device=device) * step
                d2 = nk + torch.arange(nk, device=device) * step
        else:
            # Cases 7 (L) and 8 (U): lda_rfp = nk
            if lower:
                # L11 at C+nk (upper), L22 at C+0 (lower)
                d1 = nk + torch.arange(nk, device=device) * (nk + 1)
                d2 = torch.arange(nk, device=device) * (nk + 1)
            else:
                # U11 at C+nk*(nk+1) (upper), U22 at C+nk*nk (lower)
                d1 = nk * (nk + 1) + torch.arange(nk, device=device) * (nk + 1)
                d2 = nk * nk + torch.arange(nk, device=device) * (nk + 1)

    return torch.cat([d1, d2])


def add_to_diagonal(
    C: torch.Tensor, value: float, transr: str = "T", uplo: str = "U", n: int = None
) -> None:
    """Add a scalar to every diagonal element of an RFP-packed matrix.

    This is equivalent to ``M += value * I`` on the underlying symmetric matrix,
    and is useful for regularizing before Cholesky factorization.

    Args:
        C:      RFP-packed float32 CUDA tensor of size n*(n+1)//2.
        value:  Scalar to add to each diagonal element.
        transr: RFP storage variant, 'N' or 'T' (default 'T').
        uplo:   Triangle stored, 'L' or 'U' (default 'U').
        n:      Matrix order (inferred from C.numel() if not given).
    """
    _validate(C, "C")
    if n is None:
        n = _n_from_rfp(C.numel())
    idx = rfp_diag_indices(n, transr=transr, uplo=uplo, device=C.device)
    C[idx] += value


# ---------------------------------------------------------------------------
# Handle class (public, for power users who need explicit control)
# ---------------------------------------------------------------------------
class Handle(_HandleBase):
    """curfp library handle — wraps cuBLAS and cuSOLVER handles.

    For most use cases the global handle managed by curfp is sufficient.
    Use this class only if you need multiple independent handles or fine
    control over stream assignment per handle.

    Can be used as a context manager::

        with curfp.Handle() as h:
            curfp.ssfrk_raw(h, ...)
    """

    def set_stream(self, stream: torch.cuda.Stream) -> None:
        """Bind this handle to a ``torch.cuda.Stream``."""
        self.set_stream_ptr(stream.cuda_stream)


# ---------------------------------------------------------------------------
# Low-level (raw) API — explicit handle, integer enums, all parameters
# ---------------------------------------------------------------------------
def ssfrk_raw(
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
    """Low-level symmetric rank-k update in RFP format.

    Identical to the old ``ssfrk`` API. Use integer enum constants
    (``curfp.OP_N``, ``curfp.FILL_LOWER``, etc.) and pass an explicit handle.
    """
    _validate(A, "A")
    _validate(C, "C")
    _ssfrk_C(
        handle,
        transr,
        uplo,
        trans,
        n,
        k,
        float(alpha),
        A.data_ptr(),
        lda,
        float(beta),
        C.data_ptr(),
    )


def spftrf_raw(
    handle: Handle,
    transr: int,
    uplo: int,
    n: int,
    A: torch.Tensor,
) -> int:
    """Low-level Cholesky factorization in RFP format.

    Returns info: 0 = success, >0 = leading minor of order info not positive definite.
    """
    _validate(A, "A")
    return _spftrf_C(handle, transr, uplo, n, A.data_ptr())


def spftrs_raw(
    handle: Handle,
    transr: int,
    uplo: int,
    n: int,
    nrhs: int,
    A: torch.Tensor,
    B: torch.Tensor,
    ldb: int,
) -> None:
    """Low-level triangular solve using RFP Cholesky factor.

    Solves A * X = B in-place on B.
    """
    _validate(A, "A")
    _validate(B, "B")
    _spftrs_C(handle, transr, uplo, n, nrhs, A.data_ptr(), B.data_ptr(), ldb)


# ---------------------------------------------------------------------------
# High-level API — global handle, string params, dimension inference
# ---------------------------------------------------------------------------
def ssfrk(
    A: torch.Tensor,
    C: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    transr: str = "T",
    uplo: str = "U",
    trans: str = "T",
    n: int = None,
    k: int = None,
    lda: int = None,
) -> None:
    """Symmetric rank-k update directly into RFP-packed storage.

    Computes::

        trans='T':  C := alpha * A @ A.T + beta * C   (A is row-major n×k)
        trans='N':  C := alpha * A.T @ A + beta * C   (A is col-major n×k, lda=n)

    Args:
        A:      float32 contiguous CUDA tensor.
        C:      float32 contiguous CUDA tensor of size n*(n+1)//2 (RFP format).
        alpha:  Scalar for A @ A.T term (default 1.0).
        beta:   Scalar for existing C term (default 0.0).
        transr: RFP storage variant, 'N' or 'T' (default 'T', optimal).
        uplo:   Triangle to store, 'L' or 'U' (default 'U', optimal).
        trans:  Operation on A, 'N' or 'T' (default 'T' for row-major A).
        n:      Matrix order (inferred from C.numel() if not given).
        k:      Rank of update (inferred from A.shape if not given).
        lda:    Leading dimension of A (inferred from A.shape and trans if not given).
    """
    _validate(A, "A")
    _validate(C, "C")

    if n is None:
        n = _n_from_rfp(C.numel())
    trans_upper = trans.upper()
    if k is None:
        k = A.shape[1] if trans_upper == "T" else A.shape[0]
    if lda is None:
        lda = A.shape[1] if trans_upper == "T" else A.shape[0]

    _ssfrk_C(
        _get_handle(),
        _op(transr),
        _fill(uplo),
        _op(trans),
        n,
        k,
        float(alpha),
        A.data_ptr(),
        lda,
        float(beta),
        C.data_ptr(),
    )


def spftrf(
    C: torch.Tensor,
    *,
    n: int = None,
    transr: str = "T",
    uplo: str = "U",
    check: bool = True,
) -> int:
    """In-place Cholesky factorization of an RFP-packed symmetric matrix.

    Args:
        C:      float32 contiguous CUDA tensor of size n*(n+1)//2 (RFP format).
                Overwritten with the Cholesky factor on success.
        n:      Matrix order (inferred from C.numel() if not given).
        transr: RFP storage variant, 'N' or 'T' (default 'T').
        uplo:   Triangle stored, 'L' or 'U' (default 'U').
        check:  If True (default), raise ``torch.linalg.LinAlgError`` when the
                matrix is not positive definite. If False, return info silently.

    Returns:
        info (int): 0 = success, >0 = leading minor of order info not positive definite.
                    Only meaningful when ``check=False``.
    """
    _validate(C, "C")
    if n is None:
        n = _n_from_rfp(C.numel())

    info = _spftrf_C(_get_handle(), _op(transr), _fill(uplo), n, C.data_ptr())
    if check and info != 0:
        raise torch.linalg.LinAlgError(
            f"spftrf: matrix is not positive definite "
            f"(leading minor of order {info} is not positive definite)"
        )
    return info


def spftrs(
    C: torch.Tensor,
    B: torch.Tensor,
    *,
    n: int = None,
    nrhs: int = None,
    transr: str = "T",
    uplo: str = "U",
) -> None:
    """Solve A * X = B using the RFP Cholesky factor from ``spftrf``.

    B is overwritten with the solution X in-place.

    Args:
        C:      float32 contiguous CUDA tensor, RFP Cholesky factor from ``spftrf``.
        B:      float32 contiguous CUDA tensor of shape (n, nrhs), overwritten with X.
                For a single right-hand side, shape (n,) or (n, 1) both work.
        n:      Matrix order (inferred from C.numel() if not given).
        nrhs:   Number of right-hand sides (inferred from B.shape if not given).
        transr: RFP storage variant, 'N' or 'T' — must match what was used in ssfrk/spftrf.
        uplo:   Triangle stored, 'L' or 'U' — must match what was used in ssfrk/spftrf.
    """
    _validate(C, "C")
    _validate(B, "B")

    if n is None:
        n = _n_from_rfp(C.numel())

    # Handle 1-D B by treating it as (n, 1)
    squeezed = B.ndim == 1
    if squeezed:
        B = B.unsqueeze(1)

    if nrhs is None:
        nrhs = B.shape[1]

    if B.shape[0] != n:
        raise ValueError(f"B has {B.shape[0]} rows but n={n}")

    # cuBLAS STRSM expects column-major (n, nrhs) with ldb=n.
    # PyTorch tensors are row-major, so B[i,j] is at B + i*nrhs + j, not B + j*n + i.
    # Fix: B.t().contiguous() is (nrhs, n) C-order, which in memory is exactly
    # (n, nrhs) column-major with leading dimension n.  We solve into B_f then
    # copy the result back into B in the original row-major layout.
    B_f = B.t().contiguous()  # (nrhs, n) C-order = (n, nrhs) column-major, ldb=n

    _spftrs_C(
        _get_handle(),
        _op(transr),
        _fill(uplo),
        n,
        nrhs,
        C.data_ptr(),
        B_f.data_ptr(),
        n,  # ldb = n (column-major leading dimension)
    )

    # Copy result from column-major work buffer back into the original row-major B
    B.copy_(B_f.t())

    if squeezed:
        B.squeeze_(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # High-level API
    "ssfrk",
    "spftrf",
    "spftrs",
    # Diagonal utilities
    "rfp_diag_indices",
    "add_to_diagonal",
    # Stream control
    "set_stream",
    # Low-level / power-user API
    "ssfrk_raw",
    "spftrf_raw",
    "spftrs_raw",
    "Handle",
    # Integer enum constants
    "OP_N",
    "OP_T",
    "FILL_LOWER",
    "FILL_UPPER",
]
