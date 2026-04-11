/*
 * curfpSsfr2k — Symmetric Rank-2K update in RFP format (single precision)
 *
 * Computes:
 *   trans == CURFP_OP_N:  C := alpha*A*B^T + alpha*B*A^T + beta*C  (A,B are n×k)
 *   trans == CURFP_OP_T:  C := alpha*A^T*B + alpha*B^T*A + beta*C  (A,B are k×n)
 *
 * where C is an n×n symmetric matrix in RFP format (n*(n+1)/2 floats).
 *
 * Algorithm: 2×2 block decomposition — NO workspace allocation.
 *
 * Exactly analogous to curfpSsfrk (rank-K update) but with two cross-term
 * sgemm calls on the off-diagonal S block instead of one:
 *
 *   ssyr2k(T1, A1, B1, beta)     → diagonal block 1
 *   ssyr2k(T2, A2, B2, beta)     → diagonal block 2
 *   sgemm(S, op(Ag1), op(Bg2)^T, beta)  → off-diagonal, first cross term
 *   sgemm(S, op(Bg1), op(Ag2)^T, 1.0)  → off-diagonal, second cross term (accumulate)
 *
 * Sub-block parameters are derived from the same get_ssfrk_params() table
 * used by curfpSsfrk.  A and B share the same split points and lda/ldb.
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Per-case parameters (copied from curfp_ssfrk.cpp — identical table).
 * ------------------------------------------------------------------------- */
typedef struct {
    cublasFillMode_t  fill1;
    int               dim1;
    long              off1;

    cublasFillMode_t  fill2;
    int               dim2;
    long              off2;

    int               gemm_m, gemm_n;
    long              offg;
    int               gemm_a1_first;

    long              a2_notrans;
    long              a2_trans_k;

    int               ldc;
} ssfr2k_params_t;

static void get_ssfr2k_params(int nisodd, int normaltransr, int lower,
                                int n, int n1, int n2, int nk,
                                ssfr2k_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            p->ldc = n;
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = 0;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n;
                p->gemm_m = n2; p->gemm_n = n1; p->offg = n1;
                p->gemm_a1_first = 0;
                p->a2_notrans = n1; p->a2_trans_k = n1;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = n2;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n1;
                p->gemm_m = n1; p->gemm_n = n2; p->offg = 0;
                p->gemm_a1_first = 1;
                p->a2_notrans = n2 - 1; p->a2_trans_k = n2 - 1;
            }
        } else {
            if (lower) {
                p->ldc = n1;
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = 0;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = 1;
                p->gemm_m = n1; p->gemm_n = n2; p->offg = (long)n1 * n1;
                p->gemm_a1_first = 1;
                p->a2_notrans = n1; p->a2_trans_k = n1;
            } else {
                p->ldc = n2;
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = (long)n2 * n2;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = (long)n1 * n2;
                p->gemm_m = n2; p->gemm_n = n1; p->offg = 0;
                p->gemm_a1_first = 0;
                p->a2_notrans = n1; p->a2_trans_k = n1;
            }
        }
    } else {
        if (normaltransr) {
            p->ldc = n + 1;
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = 1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = 0;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = nk + 1;
                p->gemm_a1_first = 0;
                p->a2_notrans = nk; p->a2_trans_k = nk;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = nk + 1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = nk;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = 0;
                p->gemm_a1_first = 1;
                p->a2_notrans = nk; p->a2_trans_k = nk;
            }
        } else {
            p->ldc = nk;
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = nk;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = 0;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = (long)(nk + 1) * nk;
                p->gemm_a1_first = 1;
                p->a2_notrans = nk; p->a2_trans_k = nk;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = (long)nk * (nk + 1);
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = (long)nk * nk;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = 0;
                p->gemm_a1_first = 0;
                p->a2_notrans = nk; p->a2_trans_k = nk;
            }
        }
    }
}

extern "C"
curfpStatus_t curfpSsfr2k(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpOperation_t trans,
    int              n,
    int              k,
    const float     *alpha,
    const float     *A,
    int              lda,
    const float     *B,
    int              ldb,
    const float     *beta,
    float           *C)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || k < 0 || lda < 1 || ldb < 1) return CURFP_STATUS_INVALID_VALUE;
    if (!alpha || !beta)                        return CURFP_STATUS_INVALID_VALUE;
    if (n == 0)                                 return CURFP_STATUS_SUCCESS;

    /* Quick returns */
    if ((*alpha == 0.0f || k == 0) && *beta == 1.0f)
        return CURFP_STATUS_SUCCESS;

    if (*alpha == 0.0f && *beta == 0.0f) {
        long ntotal = (long)n * (n + 1) / 2;
        CURFP_CHECK_CUDA(cudaMemset(C, 0, ntotal * sizeof(float)));
        return CURFP_STATUS_SUCCESS;
    }

    cublasHandle_t cb = handle->cublas;

    const int notrans      = (trans  == CURFP_OP_N);
    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd       = (n % 2 != 0);

    const cublasOperation_t opN  = CUBLAS_OP_N;
    const cublasOperation_t opT  = CUBLAS_OP_T;
    const cublasOperation_t opA  = notrans ? opN : opT;
    const cublasOperation_t opAt = notrans ? opT : opN;

    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        nk = n / 2; n1 = nk; n2 = nk;
    }

    ssfr2k_params_t p;
    get_ssfr2k_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    /* A and B sub-block pointers (same split for both matrices) */
    const float *A1 = A;
    const float *A2 = notrans ? A + p.a2_notrans
                              : A + p.a2_trans_k * (long)lda;
    const float *B1 = B;
    const float *B2 = notrans ? B + p.a2_notrans
                              : B + p.a2_trans_k * (long)ldb;

    /* ssyr2k on diagonal block 1 */
    CURFP_CHECK_CUBLAS(cublasSsyr2k(cb,
        p.fill1, opA,
        p.dim1, k, alpha,
        A1, lda, B1, ldb,
        beta, C + p.off1, p.ldc));

    /* ssyr2k on diagonal block 2 */
    CURFP_CHECK_CUBLAS(cublasSsyr2k(cb,
        p.fill2, opA,
        p.dim2, k, alpha,
        A2, lda, B2, ldb,
        beta, C + p.off2, p.ldc));

    /* Two sgemm calls on the off-diagonal S block.
     *
     * S stores the rectangular cross block.  The rank-2k expansion gives
     * two cross terms; gemm_a1_first controls which of (A,B) or (B,A) goes
     * as the first/second pair:
     *
     *   gemm_a1_first == 1: outer pair is (A1,A2), inner pair is (B1,B2)
     *     first  sgemm: S = alpha * op(A_outer) * op(B_inner)^T + beta*S
     *     second sgemm: S += alpha * op(B_outer) * op(A_inner)^T
     *
     *   gemm_a1_first == 0: outer pair is (A2,A1), inner is (B2,B1)
     *     first  sgemm: S = alpha * op(A_outer) * op(B_inner)^T + beta*S
     *     second sgemm: S += alpha * op(B_outer) * op(A_inner)^T
     */
    const float one = 1.0f;

    const float *Ag1 = p.gemm_a1_first ? A1 : A2;
    const float *Ag2 = p.gemm_a1_first ? A2 : A1;
    const float *Bg1 = p.gemm_a1_first ? B1 : B2;
    const float *Bg2 = p.gemm_a1_first ? B2 : B1;

    /* First cross term: alpha * op(Ag1) * op(Bg2)^T + beta*S */
    CURFP_CHECK_CUBLAS(cublasSgemm(cb,
        opA, opAt,
        p.gemm_m, p.gemm_n, k,
        alpha, Ag1, lda, Bg2, ldb,
        beta,  C + p.offg, p.ldc));

    /* Second cross term: alpha * op(Bg1) * op(Ag2)^T + 1*S (accumulate) */
    CURFP_CHECK_CUBLAS(cublasSgemm(cb,
        opA, opAt,
        p.gemm_m, p.gemm_n, k,
        alpha, Bg1, ldb, Ag2, lda,
        &one,  C + p.offg, p.ldc));

    return CURFP_STATUS_SUCCESS;
}
