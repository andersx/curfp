/*
 * curfpSpftrf — Cholesky factorization in RFP format (single precision)
 *
 * Direct translation of LAPACK spftrf.f to CUDA using cuSOLVER for SPOTRF
 * and cuBLAS for STRSM and SSYRK.
 *
 * The RFP array A (0-based, as in LAPACK spftrf which uses A(0:*)) is
 * partitioned into blocks; pointer offsets follow the LAPACK reference source
 * exactly and translate directly to C pointer arithmetic with no adjustment.
 *
 * There are 8 RFP storage variants:
 *   N parity (odd/even) × transr (N/T) × uplo (L/U)
 * Each variant performs: SPOTRF → STRSM → SSYRK → SPOTRF
 *
 * cuSOLVER's spotrf requires a workspace buffer that we allocate and free per
 * call.  A device-side devInfo integer is used to retrieve the factorization
 * status.
 */

#include <stdlib.h>
#include "curfp_internal.h"

/* Helper: run cusolverDnSpotrf on a sub-block of the RFP array.
 * Allocates workspace internally, writes result back into A_blk in place.
 * On return, *info is set: 0 = success, >0 = not positive definite. */
static curfpStatus_t run_spotrf(
    cusolverDnHandle_t solver,
    cublasFillMode_t   fill,
    int                n,
    float             *A_blk,
    int                lda,
    int               *info_out)
{
    int lwork = 0;
    CURFP_CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(solver, fill, n,
                                                      A_blk, lda, &lwork));

    float *work = NULL;
    CURFP_CHECK_CUDA(cudaMalloc((void **)&work, (size_t)lwork * sizeof(float)));

    int *devInfo = NULL;
    CURFP_CHECK_CUDA(cudaMalloc((void **)&devInfo, sizeof(int)));
    CURFP_CHECK_CUDA(cudaMemset(devInfo, 0, sizeof(int)));

    cusolverStatus_t st = cusolverDnSpotrf(solver, fill, n,
                                            A_blk, lda, work, lwork, devInfo);

    /* Copy devInfo back to host before freeing */
    int h_info = 0;
    cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(work);
    cudaFree(devInfo);

    if (st != CUSOLVER_STATUS_SUCCESS) return from_cusolver_status(st);
    *info_out = h_info;
    return CURFP_STATUS_SUCCESS;
}

curfpStatus_t curfpSpftrf(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    float           *A,
    int             *info)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0)  return CURFP_STATUS_INVALID_VALUE;
    if (!info)  return CURFP_STATUS_INVALID_VALUE;
    *info = 0;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    cublasHandle_t     cb = handle->cublas;
    cusolverDnHandle_t cs = handle->cusolver;

    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd       = (n % 2 != 0);

    /* Sub-block dimensions.  Convention follows LAPACK spftrf.f:
     *   odd N, lower:  n2=n/2, n1=n-n2  (n1=ceil(n/2))
     *   odd N, upper:  n1=n/2, n2=n-n1  (n2=ceil(n/2))
     *   even N:        k=n/2 (n1=n2=k)                     */
    int n1, n2, nk;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        nk = n / 2;
        n1 = nk; n2 = nk;   /* unused individually for even N */
    }

    const float one  =  1.0f;
    const float mone = -1.0f;

    curfpStatus_t st;
    int sub_info = 0;

    /* =====================================================================
     * 8 RFP cases
     * ===================================================================== */

    if (nisodd) {
        /* ------------------------------------------------------------------ */
        /* Odd N                                                               */
        /* ------------------------------------------------------------------ */
        if (normaltransr) {
            /* lda_rfp = N for both sub-blocks */
            if (lower) {
                /* Case 1: odd, TRANSR=N, UPLO=L
                 * Block layout (0-based offsets):
                 *   L11 at A+0  (n1 x n1, lower, lda=n)
                 *   L21 at A+n1 (n2 x n1, lda=n)
                 *   L22 at A+n  (n2 x n2, upper, lda=n)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, n1, A + 0, n, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    n2, n1, &one, A + 0, n, A + n1, n));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    n2, n1, &mone, A + n1, n, &one, A + n, n));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, n2, A + n, n, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + n1; return CURFP_STATUS_SUCCESS; }

            } else {
                /* Case 2: odd, TRANSR=N, UPLO=U
                 * Block layout (0-based offsets):
                 *   U11 at A+n2  (n1 x n1, lower, lda=n)
                 *   U21 at A+0   (n1 x n2, lda=n)   — note: stored transposed
                 *   U22 at A+n1  (n2 x n2, upper, lda=n)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, n1, A + n2, n, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    n1, n2, &one, A + n2, n, A + 0, n));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    n2, n1, &mone, A + 0, n, &one, A + n1, n));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, n2, A + n1, n, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + n1; return CURFP_STATUS_SUCCESS; }
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 3: odd, TRANSR=T, UPLO=L
                 * lda_rfp = n1
                 * Block layout (0-based offsets):
                 *   L11 at A+0      (n1 x n1, upper, lda=n1)
                 *   L21 at A+n1*n1  (n1 x n2, lda=n1)
                 *   L22 at A+1      (n2 x n2, lower, lda=n1)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, n1, A + 0, n1, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    n1, n2, &one, A + 0, n1, A + (long)n1 * n1, n1));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    n2, n1, &mone, A + (long)n1 * n1, n1, &one, A + 1, n1));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, n2, A + 1, n1, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + n1; return CURFP_STATUS_SUCCESS; }

            } else {
                /* Case 4: odd, TRANSR=T, UPLO=U
                 * lda_rfp = n2
                 * Block layout (0-based offsets):
                 *   U11 at A+n2*n2  (n1 x n1, upper, lda=n2)
                 *   U21 at A+0      (n2 x n1, lda=n2)
                 *   U22 at A+n1*n2  (n2 x n2, lower, lda=n2)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, n1, A + (long)n2 * n2, n2, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    n2, n1, &one, A + (long)n2 * n2, n2, A + 0, n2));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    n2, n1, &mone, A + 0, n2, &one, A + (long)n1 * n2, n2));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, n2, A + (long)n1 * n2, n2, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + n1; return CURFP_STATUS_SUCCESS; }
            }
        }
    } else {
        /* ------------------------------------------------------------------ */
        /* Even N: k = N/2                                                    */
        /* ------------------------------------------------------------------ */
        const int k = nk;

        if (normaltransr) {
            /* lda_rfp = N+1 */
            if (lower) {
                /* Case 5: even, TRANSR=N, UPLO=L
                 * Block layout (0-based offsets):
                 *   L11 at A+1    (k x k, lower, lda=n+1)
                 *   L21 at A+k+1  (k x k, lda=n+1)
                 *   L22 at A+0    (k x k, upper, lda=n+1)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, k, A + 1, n + 1, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    k, k, &one, A + 1, n + 1, A + k + 1, n + 1));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    k, k, &mone, A + k + 1, n + 1, &one, A + 0, n + 1));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, k, A + 0, n + 1, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + k; return CURFP_STATUS_SUCCESS; }

            } else {
                /* Case 6: even, TRANSR=N, UPLO=U
                 * Block layout (0-based offsets):
                 *   U11 at A+k+1  (k x k, lower, lda=n+1)
                 *   U21 at A+0    (k x k, lda=n+1)
                 *   U22 at A+k    (k x k, upper, lda=n+1)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, k, A + k + 1, n + 1, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    k, k, &one, A + k + 1, n + 1, A + 0, n + 1));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    k, k, &mone, A + 0, n + 1, &one, A + k, n + 1));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, k, A + k, n + 1, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + k; return CURFP_STATUS_SUCCESS; }
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 7: even, TRANSR=T, UPLO=L
                 * lda_rfp = k
                 * Block layout (0-based offsets):
                 *   L11 at A+k       (k x k, upper, lda=k)
                 *   L21 at A+k*(k+1) (k x k, lda=k)
                 *   L22 at A+0       (k x k, lower, lda=k)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, k, A + k, k, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    k, k, &one, A + k, k, A + (long)k * (k + 1), k));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    k, k, &mone, A + (long)k * (k + 1), k, &one, A + 0, k));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, k, A + 0, k, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + k; return CURFP_STATUS_SUCCESS; }

            } else {
                /* Case 8: even, TRANSR=T, UPLO=U
                 * lda_rfp = k
                 * Block layout (0-based offsets):
                 *   U11 at A+k*(k+1) (k x k, upper, lda=k)
                 *   U21 at A+0       (k x k, lda=k)
                 *   U22 at A+k*k     (k x k, lower, lda=k)
                 */
                st = run_spotrf(cs, CUBLAS_FILL_MODE_UPPER, k, A + (long)k * (k + 1), k, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

                CURFP_CHECK_CUBLAS(cublasStrsm(cb,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    k, k, &one, A + (long)k * (k + 1), k, A + 0, k));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    k, k, &mone, A + 0, k, &one, A + (long)k * k, k));

                st = run_spotrf(cs, CUBLAS_FILL_MODE_LOWER, k, A + (long)k * k, k, &sub_info);
                if (st != CURFP_STATUS_SUCCESS) return st;
                if (sub_info != 0) { *info = sub_info + k; return CURFP_STATUS_SUCCESS; }
            }
        }
    }

    return CURFP_STATUS_SUCCESS;
}
