/*
 * curfpDstrttf — Copy triangular matrix from standard full to RFP format
 *               (double precision)
 *
 * Double-precision copy of curfp_strttf.cu: float -> double,
 * sizeof(float) -> sizeof(double). Index arithmetic is identical.
 */

#include "curfp_internal.h"

#define TX 32
#define TY 8

static __global__ void k_dstrttf(
    const double * __restrict__ A,
    double       * __restrict__ arf,
    int n, int lda,
    int nisodd, int normaltransr, int lower,
    int n1, int n2, int nk, int nt)
{
    int col = blockIdx.x * TX + threadIdx.x;
    int row = blockIdx.y * TY + threadIdx.y;

    if (col >= n || row >= n) return;

    if (lower) {
        if (col > row) return;
    } else {
        if (row > col) return;
    }

    long arf_idx = -1;

    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                if (col >= n1 && row >= n2) {
                    arf_idx = (long)(row - n2) * n + (col - n1);
                } else {
                    arf_idx = (long)col * n + row;
                }
            } else {
                if (col >= n1) {
                    long ij0 = (long)nt - (long)(n - col) * n;
                    arf_idx = ij0 + row;
                } else {
                    long ij0 = (long)nt - (long)(n2 - row) * n;
                    arf_idx = ij0 + (long)(n1 + 1 + col);
                }
            }
        } else {
            if (lower) {
                if (col < n1) {
                    arf_idx = (long)row * n1 + col;
                } else {
                    arf_idx = (long)(col - n1) * n1 + (row - n1 + 1);
                }
            } else {
                if (col >= n1 && row <= n1) {
                    arf_idx = (long)row * n2 + (col - n1);
                } else if (col < n1) {
                    arf_idx = (long)(n1 + 1) * n2 + (long)col * (n1 + 1) + row;
                } else {
                    long ij1 = (long)(n1 + 1) * n2 + (long)(row - n2) * (n1 + 1);
                    arf_idx = ij1 + (row - n2 + 1) + (col - row);
                }
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                if (col >= nk) {
                    arf_idx = (long)(row - nk) * (n + 1) + (col - nk);
                } else {
                    arf_idx = (long)col * (n + 1) + row + 1;
                }
            } else {
                if (col >= nk) {
                    arf_idx = (long)nt - (long)(n + 1) * (n - col) + row;
                } else {
                    arf_idx = (long)nt - (long)(n + 1) * (nk - row) + (nk + 1 + col);
                }
            }
        } else {
            if (lower) {
                if (col == nk && row >= nk) {
                    arf_idx = row - nk;
                } else if (col < nk) {
                    arf_idx = (long)nk * (row + 1) + col;
                } else {
                    arf_idx = (long)nk * (col - nk) + (row - nk);
                }
            } else {
                if (col >= nk && row <= nk) {
                    arf_idx = (long)row * nk + (col - nk);
                } else if (col < nk) {
                    arf_idx = (long)nk * (nk + 1) + (long)col * nk + row;
                } else {
                    long ij1 = (long)nk * (nk + 1) + (long)(row - nk - 1) * nk;
                    arf_idx = ij1 + (row - nk) + (col - row);
                }
            }
        }
    }

    if (arf_idx >= 0)
        arf[arf_idx] = A[(long)row * lda + col];
}

extern "C"
curfpStatus_t curfpDstrttf(curfpHandle_t    handle,
                            curfpOperation_t transr,
                            curfpFillMode_t  uplo,
                            int              n,
                            const double    *A,
                            int              lda,
                            double          *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || lda < n) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    if (n == 1) {
        CURFP_CHECK_CUDA(cudaMemcpy(arf, A, sizeof(double),
                                    cudaMemcpyDeviceToDevice));
        return CURFP_STATUS_SUCCESS;
    }

    cudaStream_t stream;
    CURFP_CHECK_CUBLAS(cublasGetStream(handle->cublas, &stream));

    int nisodd       = n % 2;
    int normaltransr = (transr == CURFP_OP_N);
    int lower        = (uplo == CURFP_FILL_MODE_LOWER);

    int n1 = 0, n2 = 0, nk = 0;
    if (lower) { n2 = n / 2; n1 = n - n2; }
    else       { n1 = n / 2; n2 = n - n1; }
    if (!nisodd) nk = n / 2;
    int nt = n * (n + 1) / 2;

    dim3 block(TX, TY);
    dim3 grid((n + TX - 1) / TX, (n + TY - 1) / TY);

    k_dstrttf<<<grid, block, 0, stream>>>(
        A, arf, n, lda, nisodd, normaltransr, lower, n1, n2, nk, nt);

    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) return CURFP_STATUS_EXECUTION_FAILED;

    return CURFP_STATUS_SUCCESS;
}
