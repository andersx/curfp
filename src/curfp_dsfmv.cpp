/*
 * curfpDsfmv — Symmetric matrix-vector multiply in RFP format (double prec.)
 *
 * Computes:  y := alpha * A * x + beta * y
 *
 * Double-precision copy of curfp_ssfmv.cpp: float -> double,
 * cublasSsymv -> cublasDsymv, cublasSgemv -> cublasDgemv.
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
} dsfmv_params_t;

static void get_dsfmv_params(int nisodd, int normaltransr, int lower,
                              int n, int n1, int n2, int nk,
                              dsfmv_params_t *p)
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
curfpStatus_t curfpDsfmv(curfpHandle_t    handle,
                          curfpOperation_t transr,
                          curfpFillMode_t  uplo,
                          int              n,
                          const double    *alpha,
                          const double    *arf,
                          const double    *x,
                          int              incx,
                          const double    *beta,
                          double          *y,
                          int              incy)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

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

    dsfmv_params_t p;
    get_dsfmv_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const double one = 1.0;

    CURFP_CHECK_CUBLAS(cublasDsymv(cb,
        p.fill1, p.dim1, alpha,
        arf + p.off1, p.lda1,
        x, incx,
        beta, y, incy));

    const double *step4_beta = &one;

    if (p.dim2 > 0) {
        const double *S = arf + p.s_off;
        const size_t x2off = (size_t)p.dim1 * incx;
        const size_t y2off = (size_t)p.dim1 * incy;

        if (p.dim1 > 0) {
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasDgemv(cb,
                    CUBLAS_OP_T, p.dim2, p.dim1, alpha,
                    S, p.s_lda, x + x2off, incx,
                    &one, y, incy));
                CURFP_CHECK_CUBLAS(cublasDgemv(cb,
                    CUBLAS_OP_N, p.dim2, p.dim1, alpha,
                    S, p.s_lda, x, incx,
                    beta, y + y2off, incy));
            } else {
                CURFP_CHECK_CUBLAS(cublasDgemv(cb,
                    CUBLAS_OP_N, p.dim1, p.dim2, alpha,
                    S, p.s_lda, x + x2off, incx,
                    &one, y, incy));
                CURFP_CHECK_CUBLAS(cublasDgemv(cb,
                    CUBLAS_OP_T, p.dim1, p.dim2, alpha,
                    S, p.s_lda, x, incx,
                    beta, y + y2off, incy));
            }
        } else {
            step4_beta = beta;
        }
    }

    if (p.dim2 > 0) {
        CURFP_CHECK_CUBLAS(cublasDsymv(cb,
            p.fill2, p.dim2, alpha,
            arf + p.off2, p.lda2,
            x + (size_t)p.dim1 * incx, incx,
            step4_beta, y + (size_t)p.dim1 * incy, incy));
    }

    return CURFP_STATUS_SUCCESS;
}
