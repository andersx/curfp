"""
Benchmark all 8 RFP storage variants (transr x uplo x parity) for ssfrk and spftrf,
with both even and odd matrix sizes.

The 8 cases from curfp_ssfrk.cpp:
  N parity (odd/even) x transr (N/T) x uplo (L/U)
Each uses a different sub-block layout in the RFP-packed array.

trans is fixed to 'T' (row-major A), which is the natural layout for PyTorch tensors.
"""

import torch
import curfp


def cuda_time(setup_fn, fn, warmup=2, repeat=5):
    """Return median elapsed milliseconds."""
    result = None
    for _ in range(warmup):
        setup_fn()
        result = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        setup_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2], result


sizes = [4096, 8192, 16384, 32768]
k = 128

print(f"Benchmark: all RFP variants for ssfrk + spftrf, k={k}")
print(f"  transr x uplo x parity = 2x2x2 = 8 configurations per size\n")

header = (
    f"{'n':>6} {'par':>4} {'transr':>6} {'uplo':>4}"
    f"  {'ssfrk':>10} {'spftrf':>10} {'total':>10}"
)
print(header)
print("-" * len(header))

for base_n in sizes:
    for n in [base_n, base_n + 1]:  # even, then odd
        parity = "even" if n % 2 == 0 else "odd"
        for transr in ["N", "T"]:
            for uplo in ["L", "U"]:
                # trans='T' is fixed: row-major (n,k) PyTorch tensor, lda=k
                A = torch.randn(n, k, dtype=torch.float32, device="cuda") / k**0.5
                C = torch.empty(n * (n + 1) // 2, dtype=torch.float32, device="cuda")

                def do_ssfrk():
                    curfp.ssfrk(A, C, transr=transr, uplo=uplo)

                def ssfrk_then_diag():
                    do_ssfrk()
                    curfp.add_to_diagonal(C, 1.0, transr=transr, uplo=uplo)

                t_sfrk, _ = cuda_time(lambda: None, do_ssfrk)

                t_chol, info = cuda_time(
                    ssfrk_then_diag,
                    lambda: curfp.spftrf(C, transr=transr, uplo=uplo, check=False),
                )
                assert info == 0, (
                    f"Cholesky failed: n={n} transr={transr} uplo={uplo} info={info}"
                )

                print(
                    f"{n:>6} {parity:>4}"
                    f" {transr:>6}"
                    f" {uplo:>4}"
                    f"  {t_sfrk:>8.1f}ms"
                    f"  {t_chol:>8.1f}ms"
                    f"  {t_sfrk + t_chol:>8.1f}ms"
                )

                del A, C
                torch.cuda.empty_cache()

    print()  # blank line between size groups
