/*
 * curfpSsfr2 — Symmetric Rank-2 update in RFP format (single precision)
 *
 * Computes:  C := alpha * x * y^T + alpha * y * x^T + C
 *
 * where C is an n×n symmetric matrix in RFP format (n*(n+1)/2 floats).
 *
 * Algorithm: 2×2 block decomposition — NO workspace allocation.
 *
 * With x = [x1; x2] and y = [y1; y2] split at dim1:
 *
 *   T1 += alpha * x1 * y1^T + alpha * y1 * x1^T  → cublasSsyr2(T1)
 *   T2 += alpha * x2 * y2^T + alpha * y2 * x2^T  → cublasSsyr2(T2)
 *   S  += alpha * x_outer * y_inner^T             → cublasSger(S)
 *        + alpha * y_outer * x_inner^T             → cublasSger(S)
 *
 * The same 8-case sub-block layout as curfpSsfr / curfpSsfmv applies.
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Sub-block layout — identical to curfp_ssfr.cpp.
 * ------------------------------------------------------------------------- */
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
    cublasOperation_t s_op1;   /* TRANS   → S is dim2×dim1
                                  NOTRANS → S is dim1×dim2 */
} ssfr2_params_t;

static void get_ssfr2_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               ssfr2_params_t *p)
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
curfpStatus_t curfpSsfr2(curfpHandle_t    handle,
                          curfpOperation_t transr,
                          curfpFillMode_t  uplo,
                          int              n,
                          const float     *alpha,
                          const float     *x,
                          int              incx,
                          const float     *y,
                          int              incy,
                          float           *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0)    return CURFP_STATUS_INVALID_VALUE;
    if (!alpha)   return CURFP_STATUS_INVALID_VALUE;
    if (n == 0)   return CURFP_STATUS_SUCCESS;
    if (*alpha == 0.0f) return CURFP_STATUS_SUCCESS;

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

    ssfr2_params_t p;
    get_ssfr2_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    /* Step 1: T1 += alpha * x1 * y1^T + alpha * y1 * x1^T */
    CURFP_CHECK_CUBLAS(cublasSsyr2(cb,
        p.fill1, p.dim1, alpha,
        x, incx,
        y, incy,
        arf + p.off1, p.lda1));

    if (p.dim2 > 0) {
        const float *x2 = x + (size_t)p.dim1 * incx;
        const float *y2 = y + (size_t)p.dim1 * incy;
        float       *S  = arf + p.s_off;

        /* Steps 2 & 3: S += alpha * x_outer * y_inner^T
         *                 + alpha * y_outer * x_inner^T
         *
         * s_op1 == TRANS:   S is dim2×dim1
         *   outer vectors: x2/y2 (length dim2)
         *   inner vectors: x1/y1 (length dim1)
         *   ger(dim2, dim1, alpha, x2, y1, S)   → alpha * x2 * y1^T
         *   ger(dim2, dim1, alpha, y2, x1, S)   → alpha * y2 * x1^T
         *
         * s_op1 == NOTRANS: S is dim1×dim2
         *   outer vectors: x1/y1 (length dim1)
         *   inner vectors: x2/y2 (length dim2)
         *   ger(dim1, dim2, alpha, x1, y2, S)   → alpha * x1 * y2^T
         *   ger(dim1, dim2, alpha, y1, x2, S)   → alpha * y1 * x2^T
         */
        if (p.s_op1 == CUBLAS_OP_T) {
            /* S is dim2×dim1 */
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                p.dim2, p.dim1, alpha,
                x2, incx,
                y,  incy,
                S, p.s_lda));
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                p.dim2, p.dim1, alpha,
                y2, incy,
                x,  incx,
                S, p.s_lda));
        } else {
            /* S is dim1×dim2 */
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                p.dim1, p.dim2, alpha,
                x,  incx,
                y2, incy,
                S, p.s_lda));
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                p.dim1, p.dim2, alpha,
                y,  incy,
                x2, incx,
                S, p.s_lda));
        }

        /* Step 4: T2 += alpha * x2 * y2^T + alpha * y2 * x2^T */
        CURFP_CHECK_CUBLAS(cublasSsyr2(cb,
            p.fill2, p.dim2, alpha,
            x2, incx,
            y2, incy,
            arf + p.off2, p.lda2));
    }

    return CURFP_STATUS_SUCCESS;
}
