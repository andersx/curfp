/*
 * curfpDsfr — Symmetric Rank-1 update in RFP format (double precision)
 *
 * Computes:  C := alpha * x * x^T + C
 *
 * Double-precision copy of curfp_ssfr.cpp: float -> double,
 * cublasSsyr -> cublasDsyr, cublasSger -> cublasDger.
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
} dsfr_params_t;

static void get_dsfr_params(int nisodd, int normaltransr, int lower,
                              int n, int n1, int n2, int nk,
                              dsfr_params_t *p)
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
curfpStatus_t curfpDsfr(curfpHandle_t    handle,
                         curfpOperation_t transr,
                         curfpFillMode_t  uplo,
                         int              n,
                         const double    *alpha,
                         const double    *x,
                         int              incx,
                         double          *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0)    return CURFP_STATUS_INVALID_VALUE;
    if (!alpha)   return CURFP_STATUS_INVALID_VALUE;
    if (n == 0)   return CURFP_STATUS_SUCCESS;
    if (*alpha == 0.0) return CURFP_STATUS_SUCCESS;

    cublasHandle_t cb = handle->cublas;

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

    dsfr_params_t p;
    get_dsfr_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    CURFP_CHECK_CUBLAS(cublasDsyr(cb,
        p.fill1, p.dim1, alpha,
        x, incx,
        arf + p.off1, p.lda1));

    if (p.dim2 > 0) {
        const double *x2 = x + (size_t)p.dim1 * incx;
        double       *S  = arf + p.s_off;

        if (p.s_op1 == CUBLAS_OP_T) {
            CURFP_CHECK_CUBLAS(cublasDger(cb,
                p.dim2, p.dim1, alpha,
                x2, incx,
                x,  incx,
                S, p.s_lda));
        } else {
            CURFP_CHECK_CUBLAS(cublasDger(cb,
                p.dim1, p.dim2, alpha,
                x,  incx,
                x2, incx,
                S, p.s_lda));
        }

        CURFP_CHECK_CUBLAS(cublasDsyr(cb,
            p.fill2, p.dim2, alpha,
            x2, incx,
            arf + p.off2, p.lda2));
    }

    return CURFP_STATUS_SUCCESS;
}
