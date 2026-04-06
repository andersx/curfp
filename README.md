# curfp

CUDA implementation of RFP (Rectangular Full Packed) format matrix operations,
exposed as a PyTorch-friendly Python package.

## What is RFP format?

RFP stores a symmetric or triangular N×N matrix in exactly N×(N+1)/2 elements
arranged as a rectangle, avoiding the wasted space of full N×N storage while
still enabling Level 3 BLAS operations (no triangular loops). LAPACK uses this
format in routines like `ssfrk` and `spftrf`.

## Operations

| Function | Description |
|---|---|
| `ssfrk` | Symmetric rank-k update: `C := α·A·Aᵀ + β·C` (C in RFP, single precision) |
| `spftrf` | Cholesky factorization in-place: `A = L·Lᵀ` or `A = Uᵀ·U` (A in RFP, single precision) |

Both operations support all 8 RFP storage variants (N/T transr × L/U uplo ×
odd/even N) and accept a CUDA stream for asynchronous execution.

## Requirements

- CUDA toolkit (tested with CUDA 12)
- cuBLAS and cuSOLVER (included with CUDA toolkit)
- CMake >= 3.18
- Python >= 3.8
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
git clone <repo-url>
cd curfp
make
```

This creates a `.venv` and installs the package (including `torch` and `pytest`)
into it. The C library and Python extension are built automatically via
[scikit-build-core](https://scikit-build-core.readthedocs.io/).

To use a different Python version:

```bash
make PYTHON_VER=3.12
```

To activate the venv for interactive use:

```bash
source .venv/bin/activate
```

Other useful targets:

```bash
make venv       # create .venv only
make test       # run Python tests (pytest)
make test-c     # C++ tests only (ctest)
make clean      # remove build artefacts
make distclean  # clean + remove .venv
```

## Usage

### Basic example

```python
import torch
import curfp

n, k = 64, 32

# A is n×k, C is RFP-packed n×n symmetric matrix
A = torch.randn(n, k, dtype=torch.float32, device="cuda")
C = torch.zeros(n * (n + 1) // 2, dtype=torch.float32, device="cuda")

with curfp.Handle() as h:
    # C := 1.0 * A * Aᵀ + 0.0 * C  (fills lower triangle in RFP)
    curfp.ssfrk(h, curfp.OP_N, curfp.FILL_LOWER, curfp.OP_N,
                n, k, 1.0, A, n, 0.0, C)

    # Cholesky factorization of C in-place (C must be positive definite)
    info = curfp.spftrf(h, curfp.OP_N, curfp.FILL_LOWER, n, C)
    assert info == 0, f"Cholesky failed at minor {info}"
```

### With an explicit CUDA stream

```python
stream = torch.cuda.Stream()

with curfp.Handle() as h:
    h.set_stream(stream)
    with torch.cuda.stream(stream):
        curfp.ssfrk(h, curfp.OP_N, curfp.FILL_LOWER, curfp.OP_N,
                    n, k, 1.0, A, n, 0.0, C)
```

### API reference

```python
curfp.ssfrk(handle, transr, uplo, trans, n, k, alpha, A, lda, beta, C)
```

| Parameter | Type | Description |
|---|---|---|
| `handle` | `curfp.Handle` | Library handle |
| `transr` | `OP_N` / `OP_T` | RFP storage variant |
| `uplo` | `FILL_LOWER` / `FILL_UPPER` | Triangle stored in RFP |
| `trans` | `OP_N` / `OP_T` | `OP_N`: C=α·A·Aᵀ; `OP_T`: C=α·Aᵀ·A |
| `n` | int | Order of symmetric matrix C |
| `k` | int | Rank of the update |
| `alpha` | float | Scalar for A·Aᵀ |
| `A` | `torch.Tensor` | float32 CUDA, shape `(n,k)` for `OP_N` |
| `lda` | int | Leading dimension of A |
| `beta` | float | Scalar for C |
| `C` | `torch.Tensor` | float32 CUDA, RFP format, `n*(n+1)//2` elements |

```python
info = curfp.spftrf(handle, transr, uplo, n, A)
```

| Parameter | Type | Description |
|---|---|---|
| `handle` | `curfp.Handle` | Library handle |
| `transr` | `OP_N` / `OP_T` | RFP storage variant |
| `uplo` | `FILL_LOWER` / `FILL_UPPER` | Triangle stored in RFP |
| `n` | int | Order of matrix A |
| `A` | `torch.Tensor` | float32 CUDA, RFP format, `n*(n+1)//2` elements (overwritten) |
| returns `info` | int | 0 = success; >0 = leading minor of that order is not positive definite |

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `curfp.OP_N` | 0 | No transpose |
| `curfp.OP_T` | 1 | Transpose |
| `curfp.FILL_LOWER` | 0 | Lower triangle |
| `curfp.FILL_UPPER` | 1 | Upper triangle |
