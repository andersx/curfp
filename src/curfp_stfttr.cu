/*
 * curfpSstfttr — Copy triangular matrix from RFP format to standard full format
 *
 * Inverse of curfpSstrttf: direct CUDA translation of LAPACK stfttr.f.
 * ARF is n*(n+1)/2 floats on device.
 * A is an n×n row-major matrix on device (lda >= n).
 * Only the triangle specified by UPLO is written; the other triangle
 * is not touched (caller should zero-initialize A if needed).
 *
 * Unified 2D kernel: one thread per triangular element of A.
 * Exact mirror of curfp_strttf.cu with src/dst swapped:
 *   strttf: arf[arf_idx]         = A[row*lda + col]
 *   stfttr: A[row*lda + col]     = arf[arf_idx]
 *
 * The RFP index arithmetic is identical to strttf — see that file for the
 * full derivation of arf_idx from (row, col) in each of the 8 cases.
 */

#include "curfp_internal.h"

#define TX 32
#define TY 8

static __global__ void k_stfttr(
    const float * __restrict__ arf,
    float       * __restrict__ A,
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
                /* Case 1: TRANSR='N', UPLO='L' */
                if (col >= n1 && row >= n2) {
                    arf_idx = (long)(row - n2) * n + (col - n1);
                } else {
                    arf_idx = (long)col * n + row;
                }
            } else {
                /* Case 2: TRANSR='N', UPLO='U' */
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
                /* Case 3: TRANSR='T', UPLO='L' */
                if (col < n1) {
                    arf_idx = (long)row * n1 + col;
                } else {
                    arf_idx = (long)(col - n1) * n1 + (row - n1 + 1);
                }
            } else {
                /* Case 4: TRANSR='T', UPLO='U' */
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
                /* Case 5: TRANSR='N', UPLO='L' */
                if (col >= nk) {
                    arf_idx = (long)(row - nk) * (n + 1) + (col - nk);
                } else {
                    arf_idx = (long)col * (n + 1) + row + 1;
                }
            } else {
                /* Case 6: TRANSR='N', UPLO='U' */
                if (col >= nk) {
                    arf_idx = (long)nt - (long)(n + 1) * (n - col) + row;
                } else {
                    arf_idx = (long)nt - (long)(n + 1) * (nk - row) + (nk + 1 + col);
                }
            }
        } else {
            if (lower) {
                /* Case 7: TRANSR='T', UPLO='L' */
                if (col == nk && row >= nk) {
                    arf_idx = row - nk;
                } else if (col < nk) {
                    arf_idx = (long)nk * (row + 1) + col;
                } else {
                    /* col > nk */
                    arf_idx = (long)nk * (col - nk) + (row - nk);
                }
            } else {
                /* Case 8: TRANSR='T', UPLO='U' */
                if (col >= nk && row <= nk) {
                    arf_idx = (long)row * nk + (col - nk);
                } else if (col < nk) {
                    arf_idx = (long)nk * (nk + 1) + (long)col * nk + row;
                } else {
                    /* col>=nk, row>=nk+1: B-part second */
                    long ij1 = (long)nk * (nk + 1) + (long)(row - nk - 1) * nk;
                    arf_idx = ij1 + (row - nk) + (col - row);
                }
            }
        }
    }

    if (arf_idx >= 0)
        A[(long)row * lda + col] = arf[arf_idx];
}

/* ================================================================
 * Public entry point
 * ================================================================ */
extern "C"
curfpStatus_t curfpSstfttr(curfpHandle_t    handle,
                            curfpOperation_t transr,
                            curfpFillMode_t  uplo,
                            int              n,
                            const float     *arf,
                            float           *A,
                            int              lda)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || lda < n) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    if (n == 1) {
        CURFP_CHECK_CUDA(cudaMemcpy(A, arf, sizeof(float),
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

    k_stfttr<<<grid, block, 0, stream>>>(
        arf, A, n, lda, nisodd, normaltransr, lower, n1, n2, nk, nt);

    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) return CURFP_STATUS_EXECUTION_FAILED;

    return CURFP_STATUS_SUCCESS;
}
