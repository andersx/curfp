/*
 * curfpSpftrf — Cholesky factorization in RFP format (single precision)
 *
 * Direct translation of LAPACK spftrf.f to CUDA using cuSOLVER (SPOTRF)
 * and cuBLAS (STRSM, SSYRK).
 *
 * There are 8 RFP storage variants: N parity × transr × uplo.
 * Each variant performs: SPOTRF → STRSM → SSYRK → SPOTRF on sub-blocks.
 *
 * Instead of 8 copy-pasted blocks, sub-block layout is captured in a struct
 * and one generic code path runs the 4 calls.
 */

#include <stdlib.h>
#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Helper: run cusolverDnSpotrf on one sub-block.
 * Allocates/frees workspace internally per call.
 * ------------------------------------------------------------------------- */
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
    int h_info = 0;
    cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(work);
    cudaFree(devInfo);

    if (st != CUSOLVER_STATUS_SUCCESS) return from_cusolver_status(st);
    *info_out = h_info;
    return CURFP_STATUS_SUCCESS;
}

/* -------------------------------------------------------------------------
 * Per-case parameters for the 4 cuBLAS/cuSOLVER calls.
 *
 * Block layout (all offsets in elements):
 *   blk11: spotrf1 block  (upper-left factor)
 *   blk21: strsm  block   (off-diagonal)
 *   blk22: spotrf2 block  (lower-right factor)
 *
 * STRSM solves: op(blk11) * blk21 = rhs  or  blk21 * op(blk11) = rhs
 * SSYRK updates: blk22 -= blk21 * blk21^T  (or transposed)
 * ------------------------------------------------------------------------- */
typedef struct {
    /* spotrf block 1 */
    cublasFillMode_t  fill11;
    int               dim11;
    long              off11;
    int               lda11;

    /* strsm */
    cublasSideMode_t  trsm_side;
    cublasFillMode_t  trsm_fill;
    cublasOperation_t trsm_op;
    long              trsm_a_off;  /* pointer to triangular factor (blk11) */
    long              trsm_b_off;  /* pointer to rhs/solution (blk21) */
    int               trsm_m, trsm_n;
    int               trsm_lda, trsm_ldb;

    /* ssyrk */
    cublasFillMode_t  syrk_fill;
    cublasOperation_t syrk_op;
    int               syrk_n, syrk_k;
    long              syrk_a_off;  /* blk21 pointer */
    int               syrk_lda;
    long              syrk_c_off;  /* blk22 pointer */
    int               syrk_ldc;

    /* spotrf block 2 */
    cublasFillMode_t  fill22;
    int               dim22;
    long              off22;
    int               lda22;
    int               info22_offset;  /* added to sub_info for block-2 failures */
} spftrf_params_t;

static void get_spftrf_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               spftrf_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            /* lda_rfp = n for both sub-blocks */
            if (lower) {
                /* Case 1: odd, TRANSR=N, UPLO=L
                 *   L11 at A+0  (n1×n1, lower), L21 at A+n1, L22 at A+n (n2×n2, upper) */
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = 0;    p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = 0; p->trsm_b_off = n1; p->trsm_lda = n; p->trsm_ldb = n;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = n1; p->syrk_lda = n;
                p->syrk_c_off = n; p->syrk_ldc = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n;  p->lda22 = n;
                p->info22_offset = n1;
            } else {
                /* Case 2: odd, TRANSR=N, UPLO=U
                 *   U11 at A+n2 (n1×n1, lower), U21 at A+0, U22 at A+n1 (n2×n2, upper) */
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = n2;  p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = n2; p->trsm_b_off = 0; p->trsm_lda = n; p->trsm_ldb = n;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = 0; p->syrk_lda = n;
                p->syrk_c_off = n1; p->syrk_ldc = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n1; p->lda22 = n;
                p->info22_offset = n1;
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 3: odd, TRANSR=T, UPLO=L, lda_rfp=n1
                 *   L11 at A+0 (n1×n1, upper, lda=n1), L21 at A+n1*n1, L22 at A+1 (n2×n2, lower) */
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = 0; p->lda11 = n1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = 0; p->trsm_b_off = (long)n1*n1; p->trsm_lda = n1; p->trsm_ldb = n1;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = (long)n1*n1; p->syrk_lda = n1;
                p->syrk_c_off = 1; p->syrk_ldc = n1;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = 1; p->lda22 = n1;
                p->info22_offset = n1;
            } else {
                /* Case 4: odd, TRANSR=T, UPLO=U, lda_rfp=n2
                 *   U11 at A+n2*n2 (n1×n1, upper, lda=n2), U21 at A+0, U22 at A+n1*n2 (n2×n2, lower) */
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = (long)n2*n2; p->lda11 = n2;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = (long)n2*n2; p->trsm_b_off = 0; p->trsm_lda = n2; p->trsm_ldb = n2;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = 0; p->syrk_lda = n2;
                p->syrk_c_off = (long)n1*n2; p->syrk_ldc = n2;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = (long)n1*n2; p->lda22 = n2;
                p->info22_offset = n1;
            }
        }
    } else {
        /* Even N: k = nk */
        if (normaltransr) {
            /* lda_rfp = n+1 */
            if (lower) {
                /* Case 5: even, TRANSR=N, UPLO=L
                 *   L11 at A+1 (k×k, lower, lda=n+1), L21 at A+k+1, L22 at A+0 (k×k, upper) */
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = 1;    p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = 1; p->trsm_b_off = nk+1; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = nk+1; p->syrk_lda = n+1;
                p->syrk_c_off = 0; p->syrk_ldc = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = 0;    p->lda22 = n+1;
                p->info22_offset = nk;
            } else {
                /* Case 6: even, TRANSR=N, UPLO=U
                 *   U11 at A+k+1 (k×k, lower, lda=n+1), U21 at A+0, U22 at A+k (k×k, upper) */
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = nk+1; p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk+1; p->trsm_b_off = 0; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = 0; p->syrk_lda = n+1;
                p->syrk_c_off = nk; p->syrk_ldc = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = nk;   p->lda22 = n+1;
                p->info22_offset = nk;
            }
        } else {
            /* TRANSR=T, lda_rfp=nk */
            if (lower) {
                /* Case 7: even, TRANSR=T, UPLO=L
                 *   L11 at A+k (k×k, upper, lda=k), L21 at A+k*(k+1), L22 at A+0 (k×k, lower) */
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = nk;           p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk; p->trsm_b_off = (long)nk*(nk+1); p->trsm_lda = nk; p->trsm_ldb = nk;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = (long)nk*(nk+1); p->syrk_lda = nk;
                p->syrk_c_off = 0; p->syrk_ldc = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = 0;            p->lda22 = nk;
                p->info22_offset = nk;
            } else {
                /* Case 8: even, TRANSR=T, UPLO=U
                 *   U11 at A+k*(k+1) (k×k, upper, lda=k), U21 at A+0, U22 at A+k*k (k×k, lower) */
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = (long)nk*(nk+1); p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = (long)nk*(nk+1); p->trsm_b_off = 0; p->trsm_lda = nk; p->trsm_ldb = nk;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = 0; p->syrk_lda = nk;
                p->syrk_c_off = (long)nk*nk; p->syrk_ldc = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = (long)nk*nk; p->lda22 = nk;
                p->info22_offset = nk;
            }
        }
    }
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

    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        nk = n / 2; n1 = nk; n2 = nk;
    }

    spftrf_params_t p;
    get_spftrf_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const float one  =  1.0f;
    const float mone = -1.0f;
    curfpStatus_t st;
    int sub_info = 0;

    /* 1. SPOTRF on block 11 */
    st = run_spotrf(cs, p.fill11, p.dim11, A + p.off11, p.lda11, &sub_info);
    if (st != CURFP_STATUS_SUCCESS) return st;
    if (sub_info != 0) { *info = sub_info; return CURFP_STATUS_SUCCESS; }

    /* 2. STRSM: solve triangular system to update off-diagonal block */
    CURFP_CHECK_CUBLAS(cublasStrsm(cb,
        p.trsm_side, p.trsm_fill, p.trsm_op, CUBLAS_DIAG_NON_UNIT,
        p.trsm_m, p.trsm_n, &one,
        A + p.trsm_a_off, p.trsm_lda,
        A + p.trsm_b_off, p.trsm_ldb));

    /* 3. SSYRK: update block 22 using the solved off-diagonal */
    CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
        p.syrk_fill, p.syrk_op,
        p.syrk_n, p.syrk_k, &mone,
        A + p.syrk_a_off, p.syrk_lda,
        &one, A + p.syrk_c_off, p.syrk_ldc));

    /* 4. SPOTRF on block 22 */
    st = run_spotrf(cs, p.fill22, p.dim22, A + p.off22, p.lda22, &sub_info);
    if (st != CURFP_STATUS_SUCCESS) return st;
    if (sub_info != 0) { *info = sub_info + p.info22_offset; return CURFP_STATUS_SUCCESS; }

    return CURFP_STATUS_SUCCESS;
}
