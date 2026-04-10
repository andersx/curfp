# curfp

CUDA implementation of RFP (Rectangular Full Packed) format matrix operations,
exposed as a PyTorch-friendly Python package.

![Benchmark](benchmark.png)

*Benchmarked on RTX 5070 Ti (16 GB), float32, rank-1024 update + Cholesky + solve (64 rhs).
At n ≥ 56K the dense path runs out of memory while curfp continues to scale.*

## What is RFP format?

RFP stores a symmetric or triangular N×N matrix in exactly N×(N+1)/2 elements
arranged as a rectangle, avoiding the wasted space of full N×N storage while
still enabling Level 3 BLAS operations (no triangular loops). This means you
can fit **2× larger matrices** in GPU memory compared to dense storage, and
operate on matrices that would otherwise cause out-of-memory errors.

LAPACK uses this format in routines like `ssfrk`, `spftrf`, and `spftrs`.

## Operations

| Function | Description |
|---|---|
| `ssfrk` | Symmetric rank-k update: `C := alpha * A @ A.T + beta * C` (C in RFP) |
| `spftrf` | Cholesky factorization in-place: `A = L @ L.T` or `A = U.T @ U` (A in RFP) |
| `spftrs` | Triangular solve: `(L @ L.T) @ X = B` using the Cholesky factor from `spftrf` |
| `spftri` | Matrix inversion from Cholesky factor: `A := A^{-1}` in RFP |
| `slansf` | Norm of a symmetric matrix in RFP format: max-element, 1-norm, or Frobenius |
| `spfcon` | Condition number estimate from Cholesky factor: `rcond = 1 / (‖A⁻¹‖₁ · anorm)` |
| `ssfmv` | Symmetric matrix-vector multiply: `y := alpha * A * x + beta * y` (A in RFP) |
| `strttf` | Convert a full triangular matrix to RFP format |
| `stfttr` | Convert an RFP matrix to full triangular format |
| `rfp_diag_indices` | Return flat indices of diagonal elements in the RFP array |
| `add_to_diagonal` | Add a scalar to every diagonal element (e.g. regularization) |

All operations support all 8 RFP storage variants (N/T transr × L/U uplo ×
odd/even N), are validated against LAPACK, and accept a CUDA stream for
asynchronous execution.

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

## Quick start

```python
import torch
import curfp

n, k, nrhs = 4096, 128, 10

A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")

# Symmetric rank-k update: C = A @ A.T in RFP format
curfp.ssfrk(A, C)

# Regularize: C += I  (ensures positive definiteness)
curfp.add_to_diagonal(C, 1.0)

# Cholesky factorization in-place
curfp.spftrf(C)

# Solve (A @ A.T + I) @ X = B  in-place on B
curfp.spftrs(C, B)

# --- or: compute the explicit inverse ---
curfp.spftri(C)                         # C now holds (A @ A.T + I)^{-1} in RFP

# --- or: pack/unpack for interop with dense code ---
tri  = torch.triu(some_dense_matrix)
arf  = curfp.strttf(tri, uplo='U')      # dense upper triangle -> RFP
full = curfp.stfttr(arf, uplo='U')      # RFP -> dense upper triangle
```

## Memory advantage

RFP stores only the triangle of a symmetric matrix, using half the memory:

| Matrix size | Dense (full N×N) | RFP (N×(N+1)/2) | Savings |
|-------------|-----------------|------------------|---------|
| 16,384 | 1.00 GB | 0.50 GB | 50% |
| 32,768 | 4.00 GB | 2.00 GB | 50% |
| 65,536 | 16.00 GB | 8.00 GB | 50% |

On a 16 GB GPU, dense Cholesky maxes out around n ≈ 40,000 (the matrix plus
cuSOLVER workspace exceed available memory). With curfp, you can handle
n ≈ 90,000 on the same GPU.

## API reference

### High-level API (recommended)

All high-level functions manage cuBLAS/cuSOLVER handles automatically
and infer dimensions from tensor shapes.

#### `curfp.ssfrk(A, C, *, alpha=1.0, beta=0.0, transr='T', uplo='U', trans='T', n=None, k=None, lda=None)`

Symmetric rank-k update directly into RFP-packed storage.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` | required | float32 CUDA, shape `(n, k)` |
| `C` | `Tensor` | required | float32 CUDA, `n*(n+1)//2` elements (RFP) |
| `alpha` | float | 1.0 | Scalar for `A @ A.T` |
| `beta` | float | 0.0 | Scalar for existing `C` |
| `transr` | str | `'T'` | RFP storage variant: `'N'` or `'T'` |
| `uplo` | str | `'U'` | Triangle stored: `'L'` or `'U'` |
| `trans` | str | `'T'` | `'T'`: `C = alpha * A @ A.T`; `'N'`: `C = alpha * A.T @ A` |
| `n`, `k`, `lda` | int | inferred | Override inferred dimensions |

#### `curfp.spftrf(C, *, n=None, transr='T', uplo='U', check=True) -> int`

In-place Cholesky factorization of an RFP-packed symmetric matrix.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `C` | `Tensor` | required | RFP matrix, overwritten with Cholesky factor |
| `n` | int | inferred | Matrix order |
| `transr` | str | `'T'` | Must match `ssfrk` |
| `uplo` | str | `'U'` | Must match `ssfrk` |
| `check` | bool | `True` | Raise `LinAlgError` if not positive definite |

Returns `info`: 0 = success, >0 = leading minor of that order is not positive definite.

#### `curfp.spftrs(C, B, *, n=None, nrhs=None, transr='T', uplo='U')`

Solve `A @ X = B` using the RFP Cholesky factor. `B` is overwritten with `X`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `C` | `Tensor` | required | RFP Cholesky factor from `spftrf` |
| `B` | `Tensor` | required | `(n, nrhs)` right-hand sides, overwritten with solution |
| `n` | int | inferred | Matrix order |
| `nrhs` | int | inferred | Number of right-hand sides |
| `transr` | str | `'T'` | Must match previous calls |
| `uplo` | str | `'U'` | Must match previous calls |

#### `curfp.slansf(C, norm='1', *, transr='T', uplo='U', n=None) -> float`

Compute a norm of a symmetric matrix stored in RFP format.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `C` | `Tensor` | required | float32 CUDA, `n*(n+1)//2` elements (RFP, **before** `spftrf`) |
| `norm` | str | `'1'` | Norm type: `'M'` max element, `'1'`/`'O'`/`'I'` 1-norm, `'F'`/`'E'` Frobenius |
| `transr` | str | `'T'` | RFP storage variant |
| `uplo` | str | `'U'` | Triangle stored |

Returns the requested norm value. `slansf` is the natural way to compute `anorm` before calling `spfcon`:

```python
arf = curfp.strttf(tri, uplo='U')
anorm = curfp.slansf(arf, '1')      # 1-norm before factorization
curfp.spftrf(arf)
rcond = curfp.spfcon(arf, anorm)
```

#### `curfp.spfcon(C, anorm, *, transr='T', uplo='U', n=None) -> float`

Estimate the reciprocal condition number from the RFP Cholesky factor produced by `spftrf`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `C` | `Tensor` | required | RFP Cholesky factor from `spftrf` |
| `anorm` | float | required | 1-norm of the original matrix before factorization |
| `transr` | str | `'T'` | Must match previous calls |
| `uplo` | str | `'U'` | Must match previous calls |

Returns `rcond`: estimated reciprocal condition number. A value near 1 indicates a well-conditioned matrix; near 0 indicates near-singularity. To estimate `anorm`, compute `torch.linalg.norm(A_dense, 1)` before converting to RFP format.

#### `curfp.spftri(C, *, n=None, transr='T', uplo='U')`

Compute `A^{-1}` in-place from the RFP Cholesky factor produced by `spftrf`.
After this call `C` holds the inverse of the original SPD matrix, in the same
RFP storage convention.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `C` | `Tensor` | required | RFP Cholesky factor from `spftrf`, overwritten with `A^{-1}` |
| `n` | int | inferred | Matrix order |
| `transr` | str | `'T'` | Must match previous calls |
| `uplo` | str | `'U'` | Must match previous calls |

#### `curfp.ssfmv(arf, x, y=None, *, alpha=1.0, beta=0.0, transr='T', uplo='U', n=None) -> Tensor`

Symmetric matrix-vector multiply: `y := alpha * A * x + beta * y`.
Operates directly on the RFP sub-blocks — no workspace allocation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `arf` | `Tensor` | required | float32 CUDA, `n*(n+1)//2` elements (RFP) |
| `x` | `Tensor` | required | float32 CUDA, length `n` |
| `y` | `Tensor` | `None` | float32 CUDA, length `n` (created as zeros if not given) |
| `alpha` | float | 1.0 | Scalar for `A * x` |
| `beta` | float | 0.0 | Scalar for existing `y` |
| `transr` | str | `'T'` | RFP storage variant |
| `uplo` | str | `'U'` | Triangle stored |

Returns the `y` tensor.

#### `curfp.strttf(A, *, transr='T', uplo='U', n=None) -> Tensor`

Convert a full triangular matrix to RFP format. Reads the triangle of `A`
specified by `uplo` and returns a new 1-D tensor of size `n*(n+1)//2`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` | required | float32 CUDA, shape `(n, n)` |
| `transr` | str | `'T'` | RFP storage variant |
| `uplo` | str | `'U'` | Which triangle of `A` to read |

#### `curfp.stfttr(arf, *, transr='T', uplo='U', n=None) -> Tensor`

Convert an RFP matrix to full triangular format. Only the triangle specified
by `uplo` is written; the other triangle is zero.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `arf` | `Tensor` | required | float32 CUDA, `n*(n+1)//2` elements (RFP) |
| `transr` | str | `'T'` | RFP storage variant |
| `uplo` | str | `'U'` | Which triangle to write in the output |

#### `curfp.rfp_diag_indices(n, transr='T', uplo='U', device=None) -> Tensor`

Return the flat indices of the `n` diagonal elements in the RFP array.

#### `curfp.add_to_diagonal(C, value, transr='T', uplo='U', n=None)`

Add a scalar to every diagonal element of an RFP-packed matrix.
Equivalent to `M += value * I` on the underlying symmetric matrix.

#### `curfp.set_stream(stream)`

Bind all curfp operations on the current device to a CUDA stream.
Pass `None` to revert to the default stream.

### Low-level API

For power users who need explicit handle management and full parameter control,
the raw LAPACK-style API is also available:

```python
with curfp.Handle() as h:
    curfp.ssfrk_raw(h, curfp.OP_T, curfp.FILL_UPPER, curfp.OP_T,
                    n, k, 1.0, A, k, 0.0, C)
    anorm = curfp.slansf_raw(h, curfp.NORM_ONE, curfp.OP_T, curfp.FILL_UPPER, n, C)
    info = curfp.spftrf_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, C)
    curfp.spftrs_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, nrhs, C, B, n)
    curfp.spftri_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, C)
    rcond = curfp.spfcon_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, C, anorm)
    curfp.ssfmv_raw(h, curfp.OP_T, curfp.FILL_UPPER, n,
                    1.0, arf, x, 1, 0.0, y, 1)
    curfp.strttf_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, A, n, arf)
    curfp.stfttr_raw(h, curfp.OP_T, curfp.FILL_UPPER, n, arf, A, n)
```

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `curfp.OP_N` | 0 | No transpose |
| `curfp.OP_T` | 1 | Transpose |
| `curfp.FILL_LOWER` | 0 | Lower triangle |
| `curfp.FILL_UPPER` | 1 | Upper triangle |
| `curfp.NORM_MAX` | 0 | Max-element norm |
| `curfp.NORM_ONE` | 1 | 1-norm |
| `curfp.NORM_FRO` | 2 | Frobenius norm |

## Testing

All operations are validated against LAPACK (via scipy) for correctness across
all 8 RFP storage variants, with matrix sizes from 1 to 257 (both even and odd).

```bash
make test                                        # all Python tests
python python_tests/test_vs_lapack.py            # ssfrk / spftrf / spftrs vs LAPACK
python python_tests/test_stfttr_strttf.py        # format conversion vs LAPACK
python python_tests/test_spftri.py               # inversion vs numpy.linalg.inv
python python_tests/test_ssfmv.py                # matrix-vector vs numpy
python python_tests/test_slansf.py               # matrix norms vs numpy
python python_tests/test_spfcon.py               # condition number vs scipy spocon
```
