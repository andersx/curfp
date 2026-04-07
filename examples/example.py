"""
Basic curfp example: rank-k update + regularize + Cholesky + solve.

Demonstrates the high-level API where handles, dimensions, and RFP
diagonal indexing are all managed automatically.
"""

import torch
import curfp


def cuda_time(fn):
    """Run fn() between two CUDA events and return elapsed milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


n, k, nrhs = 4096, 128, 10

# A is n×k, C is RFP-packed n×n symmetric matrix, B is n×nrhs
A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")
B = torch.randn(n, nrhs, dtype=torch.float32, device="cuda")
B_orig = B.clone()

# 1. Symmetric rank-k update: C = A @ A.T in RFP format
t_ssfrk, _ = cuda_time(lambda: curfp.ssfrk(A, C))

# 2. Regularize: C += I  (ensures positive definiteness)
t_diag, _ = cuda_time(lambda: curfp.add_to_diagonal(C, 1.0))

# 3. Cholesky factorization in-place
t_chol, _ = cuda_time(lambda: curfp.spftrf(C))

# 4. Solve (A @ A.T + I) @ X = B  in-place on B
t_solve, _ = cuda_time(lambda: curfp.spftrs(C, B))

# Verify residual
M = A @ A.t()
M.diagonal().add_(1.0)
residual = (M @ B - B_orig).norm() / B_orig.norm()

full_bytes = n * n * A.element_size()
rfp_bytes = C.numel() * C.element_size()

print(f"Solved {n}x{n} system with {nrhs} right-hand sides.")
print(f"  Relative residual: {residual:.2e}")
print()
print(f"  Input A:          {n}x{k}, {A.numel() * A.element_size() / 1024:.1f} KB")
print(f"  Full matrix:      {n}x{n}, {full_bytes / 1024**3:.3f} GB")
print(
    f"  RFP-packed (C):   {C.numel()} floats, {rfp_bytes / 1024**3:.3f} GB  ({100 * rfp_bytes / full_bytes:.1f}% of full)"
)
print()
print(f"Timings:")
print(f"  ssfrk (rank-{k} update):   {t_ssfrk:.1f} ms")
print(f"  add_to_diagonal:           {t_diag:.1f} ms")
print(f"  spftrf (Cholesky):         {t_chol:.1f} ms")
print(f"  spftrs (solve {nrhs} rhs):  {t_solve:.1f} ms")
print(f"  total:                     {t_ssfrk + t_diag + t_chol + t_solve:.1f} ms")
