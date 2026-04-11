/*
 * curfpDsfmm — Symmetric matrix-matrix multiply in RFP format (double prec.)
 *
 * Double-precision copy of curfp_ssfmm.cpp: float -> double,
 * cublasSsymm -> cublasDsymm, cublasSgemm -> cublasDgemm.
 */

#include "curfp_internal.h"

typedef struct {
    cublasFillMode_t  fill1;
    int               dim1;
    long              off1;
    int               lda1;

    cublasFillMode_t  fill2;
    int               dim2;
    long              off2;
    int               lda2;

    long              s_off;
    int               s_lda;
    cublasOperation_t s_op1;
} dsfmm_params_t;

static void get_dsfmm_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               dsfmm_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = 0;  p->lda1 = n;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n;  p->lda2 = n;
                p->s_off = n1; p->s_lda = n; p->s_op1 = CUBLAS_OP_T;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = n2; p->lda1 = n;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n1; p->lda2 = n;
                p->s_off = 0;  p->s_lda = n; p->s_op1 = CUBLAS_OP_N;
            }
        } else {
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = 0;          p->lda1 = n1;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = 1;          p->lda2 = n1;
                p->s_off = (long)n1*n1; p->s_lda = n1; p->s_op1 = CUBLAS_OP_N;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = (long)n2*n2;  p->lda1 = n2;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = (long)n1*n2;  p->lda2 = n2;
                p->s_off = 0; p->s_lda = n2; p->s_op1 = CUBLAS_OP_T;
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = 1;    p->lda1 = n+1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = 0;    p->lda2 = n+1;
                p->s_off = nk+1; p->s_lda = n+1; p->s_op1 = CUBLAS_OP_T;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = nk+1; p->lda1 = n+1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = nk;   p->lda2 = n+1;
                p->s_off = 0; p->s_lda = n+1; p->s_op1 = CUBLAS_OP_N;
            }
        } else {
            if (lower) {
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = nk;           p->lda1 = nk;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = 0;            p->lda2 = nk;
                p->s_off = (long)nk*(nk+1); p->s_lda = nk; p->s_op1 = CUBLAS_OP_N;
            } else {
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = (long)nk*(nk+1); p->lda1 = nk;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = (long)nk*nk;     p->lda2 = nk;
                p->s_off = 0; p->s_lda = nk; p->s_op1 = CUBLAS_OP_T;
            }
        }
    }
}

extern "C"
curfpStatus_t curfpDsfmm(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpSideMode_t  side,
    int              m,
    int              n,
    const double    *alpha,
    const double    *arf,
    const double    *B,
    int              ldb,
    const double    *beta,
    double          *C,
    int              ldc)
{
    CURFP_CHECK_HANDLE(handle);
    if (m < 0 || n < 0 || ldb < 1 || ldc < 1) return CURFP_STATUS_INVALID_VALUE;
    if (!alpha || !beta)                        return CURFP_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0)                       return CURFP_STATUS_SUCCESS;

    cublasHandle_t cb = handle->cublas;

    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int left         = (side   == CURFP_SIDE_LEFT);

    const int n_A   = left ? m : n;
    const int nisodd = (n_A % 2 != 0);

    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n_A / 2; n1 = n_A - n2; }
        else       { n1 = n_A / 2; n2 = n_A - n1; }
    } else {
        nk = n_A / 2; n1 = nk; n2 = nk;
    }

    dsfmm_params_t p;
    get_dsfmm_params(nisodd, normaltransr, lower, n_A, n1, n2, nk, &p);

    const double  one = 1.0;
    const double *S   = arf + p.s_off;

    if (left) {
        const double *B2 = B + p.dim1;
        double       *C2 = C + p.dim1;

        CURFP_CHECK_CUBLAS(cublasDsymm(cb,
            CUBLAS_SIDE_LEFT, p.fill1,
            p.dim1, n,
            alpha, arf + p.off1, p.lda1,
            B, ldb,
            beta, C, ldc));

        if (p.dim2 > 0) {
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    p.dim1, n, p.dim2,
                    alpha, S, p.s_lda, B2, ldb,
                    &one, C, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    p.dim1, n, p.dim2,
                    alpha, S, p.s_lda, B2, ldb,
                    &one, C, ldc));
            }

            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    p.dim2, n, p.dim1,
                    alpha, S, p.s_lda, B, ldb,
                    beta, C2, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    p.dim2, n, p.dim1,
                    alpha, S, p.s_lda, B, ldb,
                    beta, C2, ldc));
            }

            CURFP_CHECK_CUBLAS(cublasDsymm(cb,
                CUBLAS_SIDE_LEFT, p.fill2,
                p.dim2, n,
                alpha, arf + p.off2, p.lda2,
                B2, ldb,
                &one, C2, ldc));
        }

    } else {
        const double *B2 = B + (size_t)p.dim1 * ldb;
        double       *C2 = C + (size_t)p.dim1 * ldc;

        CURFP_CHECK_CUBLAS(cublasDsymm(cb,
            CUBLAS_SIDE_RIGHT, p.fill1,
            m, p.dim1,
            alpha, arf + p.off1, p.lda1,
            B, ldb,
            beta, C, ldc));

        if (p.dim2 > 0) {
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, p.dim1, p.dim2,
                    alpha, B2, ldb, S, p.s_lda,
                    &one, C, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, p.dim1, p.dim2,
                    alpha, B2, ldb, S, p.s_lda,
                    &one, C, ldc));
            }

            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, p.dim2, p.dim1,
                    alpha, B, ldb, S, p.s_lda,
                    beta, C2, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasDgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, p.dim2, p.dim1,
                    alpha, B, ldb, S, p.s_lda,
                    beta, C2, ldc));
            }

            CURFP_CHECK_CUBLAS(cublasDsymm(cb,
                CUBLAS_SIDE_RIGHT, p.fill2,
                m, p.dim2,
                alpha, arf + p.off2, p.lda2,
                B2, ldb,
                &one, C2, ldc));
        }
    }

    return CURFP_STATUS_SUCCESS;
}
