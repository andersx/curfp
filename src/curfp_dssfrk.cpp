/*
 * curfpDsfrk — Symmetric Rank-K update in RFP format (double precision)
 *
 * Double-precision copy of curfp_ssfrk.cpp: float -> double,
 * cublasSsyrk -> cublasDsyrk, cublasSgemm -> cublasDgemm.
 */

#include "curfp_internal.h"

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
} dsfrk_params_t;

static void get_dsfrk_params(int nisodd, int normaltransr, int lower,
                              int n, int n1, int n2, int nk,
                              dsfrk_params_t *p)
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

curfpStatus_t curfpDsfrk(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpOperation_t trans,
    int              n,
    int              k,
    const double    *alpha,
    const double    *A,
    int              lda,
    const double    *beta,
    double          *C)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || k < 0 || lda < 1) return CURFP_STATUS_INVALID_VALUE;
    if (!alpha || !beta)            return CURFP_STATUS_INVALID_VALUE;
    if (n == 0)                     return CURFP_STATUS_SUCCESS;

    if ((*alpha == 0.0 || k == 0) && *beta == 1.0)
        return CURFP_STATUS_SUCCESS;

    if (*alpha == 0.0 && *beta == 0.0) {
        long ntotal = (long)n * (n + 1) / 2;
        CURFP_CHECK_CUDA(cudaMemset(C, 0, ntotal * sizeof(double)));
        return CURFP_STATUS_SUCCESS;
    }

    cublasHandle_t cb = handle->cublas;

    const int notrans      = (trans  == CURFP_OP_N);
    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd       = (n % 2 != 0);

    const cublasOperation_t opN = CUBLAS_OP_N;
    const cublasOperation_t opT = CUBLAS_OP_T;
    const cublasOperation_t opA = notrans ? opN : opT;
    const cublasOperation_t opAt = notrans ? opT : opN;

    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        nk = n / 2; n1 = nk; n2 = nk;
    }

    dsfrk_params_t p;
    get_dsfrk_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const double *A1 = A;
    const double *A2 = notrans ? A + p.a2_notrans
                               : A + p.a2_trans_k * (long)lda;

    CURFP_CHECK_CUBLAS(cublasDsyrk(cb,
        p.fill1, opA,
        p.dim1, k, alpha, A1, lda, beta, C + p.off1, p.ldc));

    CURFP_CHECK_CUBLAS(cublasDsyrk(cb,
        p.fill2, opA,
        p.dim2, k, alpha, A2, lda, beta, C + p.off2, p.ldc));

    const double *Ag1 = p.gemm_a1_first ? A1 : A2;
    const double *Ag2 = p.gemm_a1_first ? A2 : A1;
    CURFP_CHECK_CUBLAS(cublasDgemm(cb,
        opA, opAt,
        p.gemm_m, p.gemm_n, k,
        alpha, Ag1, lda, Ag2, lda,
        beta,  C + p.offg, p.ldc));

    return CURFP_STATUS_SUCCESS;
}
