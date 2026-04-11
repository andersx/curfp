/*
 * curfpSsfr — Symmetric Rank-1 update in RFP format (single precision)
 *
 * Computes:  C := alpha * x * x^T + C
 *
 * where C is an n×n symmetric matrix in RFP format (n*(n+1)/2 floats).
 *
 * Algorithm: 2×2 block decomposition — NO workspace allocation.
 *
 * The RFP array contains two symmetric diagonal blocks T1, T2 and one
 * off-diagonal block S, exactly as in curfpSsfmv.  The rank-1 update
 * expands to:
 *
 *   T1 += alpha * x1 * x1^T   → cublasSsyr(T1)
 *   T2 += alpha * x2 * x2^T   → cublasSsyr(T2)
 *   S  += alpha * x_outer * x_inner^T  → cublasSger(S)
 *
 * where x = [x1; x2] split at dim1.  The ger outer/inner vectors and
 * their op depend on which case S is stored as (dim2×dim1 or dim1×dim2).
 *
 * The 8-case sub-block layout is identical to curfpSsfmv; we reuse the
 * same ssfmv_params_t struct and get_ssfmv_params() function verbatim.
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Sub-block layout (copied from curfp_ssfmv.cpp — same 8-case table).
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
} ssfr_params_t;

static void get_ssfr_params(int nisodd, int normaltransr, int lower,
                              int n, int n1, int n2, int nk,
                              ssfr_params_t *p)
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
curfpStatus_t curfpSsfr(curfpHandle_t    handle,
                         curfpOperation_t transr,
                         curfpFillMode_t  uplo,
                         int              n,
                         const float     *alpha,
                         const float     *x,
                         int              incx,
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

    ssfr_params_t p;
    get_ssfr_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    /* Step 1: T1 += alpha * x1 * x1^T */
    CURFP_CHECK_CUBLAS(cublasSsyr(cb,
        p.fill1, p.dim1, alpha,
        x, incx,
        arf + p.off1, p.lda1));

    if (p.dim2 > 0) {
        const float *x2 = x + (size_t)p.dim1 * incx;
        float       *S  = arf + p.s_off;

        /* Step 2: S += alpha * x_outer * x_inner^T
         *
         * s_op1 == TRANS:   S is dim2×dim1
         *   ger(dim2, dim1, alpha, x[dim1:], x[0:dim1], S)
         *   → outer = x2 (length dim2), inner = x1 (length dim1)
         *
         * s_op1 == NOTRANS: S is dim1×dim2
         *   ger(dim1, dim2, alpha, x[0:dim1], x[dim1:], S)
         *   → outer = x1 (length dim1), inner = x2 (length dim2)
         */
        if (p.s_op1 == CUBLAS_OP_T) {
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                p.dim2, p.dim1, alpha,
                x2, incx,
                x,  incx,
                S, p.s_lda));
        } else {
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                p.dim1, p.dim2, alpha,
                x,  incx,
                x2, incx,
                S, p.s_lda));
        }

        /* Step 3: T2 += alpha * x2 * x2^T */
        CURFP_CHECK_CUBLAS(cublasSsyr(cb,
            p.fill2, p.dim2, alpha,
            x2, incx,
            arf + p.off2, p.lda2));
    }

    return CURFP_STATUS_SUCCESS;
}
