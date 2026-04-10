/*
 * curfpSsfmv — Symmetric matrix-vector multiply in RFP format (single prec.)
 *
 * Computes:
 *   y := alpha * A * x + beta * y
 *
 * where A is an n×n symmetric matrix in RFP format (n*(n+1)/2 floats).
 *
 * Algorithm: 2×2 block decomposition — NO workspace allocation.
 *
 * The RFP array contains two symmetric diagonal blocks T1, T2 and one
 * off-diagonal block S:
 *
 *   A = [[T1,   S^T],     x = [x1, x2]
 *        [S,    T2  ]]    y = [y1, y2]
 *
 * Expands to 4 cuBLAS calls:
 *   1. SSYMV(T1):  y1  = alpha*T1*x1  + beta*y1   (carries beta for y1)
 *   2. SGEMV(S):   y1 += alpha*S'*x2              (accumulate into y1)
 *   3. SGEMV(S):   y2  = alpha*S''*x1 + beta*y2   (carries beta for y2)
 *   4. SSYMV(T2):  y2 += alpha*T2*x2              (accumulate into y2)
 *
 * Sub-block offsets, lda values, fill modes, and SGEMV operation types are
 * encoded in a params struct, one set per RFP storage variant (8 total).
 * The same 8 cases and n1/n2/k conventions used by spftrf/spftrs apply here.
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Per-case parameters.
 *
 * T1 block (diagonal): n1×n1, acts on x[0:dim1] / y[0:dim1]
 * T2 block (diagonal): n2×n2, acts on x[dim1:n] / y[dim1:n]
 * S  block (off-diag): connects the two halves
 *
 * SGEMV call ordering:
 *   s_op1 == TRANS:
 *     step2: SGEMV(TRANS,   dim2, dim1, alpha, S, s_lda, x+dim1, 1, one,  y,      1)
 *     step3: SGEMV(NOTRANS, dim2, dim1, alpha, S, s_lda, x,      1, beta, y+dim1, 1)
 *   s_op1 == NOTRANS:
 *     step2: SGEMV(NOTRANS, dim1, dim2, alpha, S, s_lda, x+dim1, 1, one,  y,      1)
 *     step3: SGEMV(TRANS,   dim1, dim2, alpha, S, s_lda, x,      1, beta, y+dim1, 1)
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
    cublasOperation_t s_op1;   /* TRANS or NOTRANS; step3 uses opposite */
} ssfmv_params_t;

static void get_ssfmv_params(int nisodd, int normaltransr, int lower,
                              int n, int n1, int n2, int nk,
                              ssfmv_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                /* Case 1: odd, TRANSR=N, UPLO=L, lda_rfp=n
                 * T1(n1×n1 LOWER) at 0, S(n2×n1) at n1, T2(n2×n2 UPPER) at n */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = 0;  p->lda1 = n;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n;  p->lda2 = n;
                p->s_off = n1; p->s_lda = n; p->s_op1 = CUBLAS_OP_T;
            } else {
                /* Case 2: odd, TRANSR=N, UPLO=U, lda_rfp=n
                 * T1(n1×n1 LOWER) at n2, S(n1×n2) at 0, T2(n2×n2 UPPER) at n1 */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = n2; p->lda1 = n;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n1; p->lda2 = n;
                p->s_off = 0;  p->s_lda = n; p->s_op1 = CUBLAS_OP_N;
            }
        } else {
            if (lower) {
                /* Case 3: odd, TRANSR=T, UPLO=L, lda_rfp=n1
                 * T1(n1×n1 UPPER) at 0, S(n1×n2) at n1², T2(n2×n2 LOWER) at 1 */
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = 0;          p->lda1 = n1;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = 1;          p->lda2 = n1;
                p->s_off = (long)n1*n1; p->s_lda = n1; p->s_op1 = CUBLAS_OP_N;
            } else {
                /* Case 4: odd, TRANSR=T, UPLO=U, lda_rfp=n2
                 * T1(n1×n1 UPPER) at n2², S(n2×n1) at 0, T2(n2×n2 LOWER) at n1*n2 */
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = (long)n2*n2;  p->lda1 = n2;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = (long)n1*n2;  p->lda2 = n2;
                p->s_off = 0; p->s_lda = n2; p->s_op1 = CUBLAS_OP_T;
            }
        }
    } else {
        /* Even N: k = nk */
        if (normaltransr) {
            if (lower) {
                /* Case 5: even, TRANSR=N, UPLO=L, lda_rfp=n+1
                 * T1(k×k LOWER) at 1, S(k×k) at k+1, T2(k×k UPPER) at 0 */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = 1;    p->lda1 = n+1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = 0;    p->lda2 = n+1;
                p->s_off = nk+1; p->s_lda = n+1; p->s_op1 = CUBLAS_OP_T;
            } else {
                /* Case 6: even, TRANSR=N, UPLO=U, lda_rfp=n+1
                 * T1(k×k LOWER) at k+1, S(k×k) at 0, T2(k×k UPPER) at k */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = nk+1; p->lda1 = n+1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = nk;   p->lda2 = n+1;
                p->s_off = 0; p->s_lda = n+1; p->s_op1 = CUBLAS_OP_N;
            }
        } else {
            if (lower) {
                /* Case 7: even, TRANSR=T, UPLO=L, lda_rfp=k
                 * T1(k×k UPPER) at k, T2(k×k LOWER) at 0, S(k×k) at k(k+1)
                 * S = TRANSR transpose of Case 5's S (A21), so here S = A12.
                 * y1 += A12 * x2 = S * x2 → NOTRANS */
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = nk;           p->lda1 = nk;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = 0;            p->lda2 = nk;
                p->s_off = (long)nk*(nk+1); p->s_lda = nk; p->s_op1 = CUBLAS_OP_N;
            } else {
                /* Case 8: even, TRANSR=T, UPLO=U, lda_rfp=k
                 * T1(k×k UPPER) at k(k+1), T2(k×k LOWER) at k², S(k×k) at 0
                 * S = TRANSR transpose of Case 6's S (A12), so here S = A21.
                 * y1 += A12 * x2 = S^T * x2 → TRANS */
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = (long)nk*(nk+1); p->lda1 = nk;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = (long)nk*nk;     p->lda2 = nk;
                p->s_off = 0; p->s_lda = nk; p->s_op1 = CUBLAS_OP_T;
            }
        }
    }
}

extern "C"
curfpStatus_t curfpSsfmv(curfpHandle_t    handle,
                          curfpOperation_t transr,
                          curfpFillMode_t  uplo,
                          int              n,
                          const float     *alpha,
                          const float     *arf,
                          const float     *x,
                          int              incx,
                          const float     *beta,
                          float           *y,
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

    ssfmv_params_t p;
    get_ssfmv_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const float one = 1.0f;

    /* Step 1: y[0:dim1] = alpha * T1 * x[0:dim1] + beta * y[0:dim1] */
    CURFP_CHECK_CUBLAS(cublasSsymv(cb,
        p.fill1, p.dim1, alpha,
        arf + p.off1, p.lda1,
        x, incx,
        beta, y, incy));

    /* Steps 2 & 3: off-diagonal S updates both halves of y.
     *
     * s_op1 == TRANS:
     *   step2: y[0:dim1] += alpha * S^T(dim1×dim2) * x[dim1:]
     *          → SGEMV(TRANS, dim2, dim1, alpha, S, lda, x+dim1, incx, one, y, incy)
     *   step3: y[dim1:]  =  alpha * S(dim2×dim1)  * x[0:dim1] + beta*y[dim1:]
     *          → SGEMV(NOTRANS, dim2, dim1, alpha, S, lda, x, incx, beta, y+dim1, incy)
     *
     * s_op1 == NOTRANS:
     *   step2: y[0:dim1] += alpha * S(dim1×dim2)  * x[dim1:]
     *          → SGEMV(NOTRANS, dim1, dim2, alpha, S, lda, x+dim1, incx, one, y, incy)
     *   step3: y[dim1:]  =  alpha * S^T(dim2×dim1) * x[0:dim1] + beta*y[dim1:]
     *          → SGEMV(TRANS, dim1, dim2, alpha, S, lda, x, incx, beta, y+dim1, incy)
     */
    /* Steps 2, 3, 4: off-diagonal S and second diagonal block T2.
     *
     * beta must be applied to y[dim1:n] exactly once.  The SGEMV in step 3
     * carries beta, but only if its leading dimension (m) is non-zero.
     *
     * TRANS path (s_op1=TRANS, S is dim2×dim1):
     *   step2: SGEMV(TRANS,   dim2, dim1, alpha, S, ..., x+dim1, ..., one,  y,      ...)
     *   step3: SGEMV(NOTRANS, dim2, dim1, alpha, S, ..., x,      ..., beta, y+dim1, ...)
     *   m=dim2 in step3 → beta always applied (dim2>0 guaranteed here).
     *
     * NOTRANS path (s_op1=NOTRANS, S is dim1×dim2):
     *   step2: SGEMV(NOTRANS, dim1, dim2, alpha, S, ..., x+dim1, ..., one,  y,      ...)
     *   step3: SGEMV(TRANS,   dim1, dim2, alpha, S, ..., x,      ..., beta, y+dim1, ...)
     *   m=dim1 in step3 → if dim1==0 (n=1, UPLO=U), SGEMV is a no-op and
     *   beta is NOT applied.  In that case carry beta in step4's SSYMV.       */

    const float *step4_beta = &one;  /* overridden below when needed */

    if (p.dim2 > 0) {
        const float *S = arf + p.s_off;
        const size_t x2off = (size_t)p.dim1 * incx;
        const size_t y2off = (size_t)p.dim1 * incy;

        if (p.dim1 > 0) {
            /* Both TRANS and NOTRANS paths are safe: the dimension feeding
             * step3's m is non-zero so SGEMV will honour the beta argument.  */
            if (p.s_op1 == CUBLAS_OP_T) {
                /* S is dim2×dim1. */
                CURFP_CHECK_CUBLAS(cublasSgemv(cb,
                    CUBLAS_OP_T, p.dim2, p.dim1, alpha,
                    S, p.s_lda, x + x2off, incx,
                    &one, y, incy));
                CURFP_CHECK_CUBLAS(cublasSgemv(cb,
                    CUBLAS_OP_N, p.dim2, p.dim1, alpha,
                    S, p.s_lda, x, incx,
                    beta, y + y2off, incy));
            } else {
                /* S is dim1×dim2. */
                CURFP_CHECK_CUBLAS(cublasSgemv(cb,
                    CUBLAS_OP_N, p.dim1, p.dim2, alpha,
                    S, p.s_lda, x + x2off, incx,
                    &one, y, incy));
                CURFP_CHECK_CUBLAS(cublasSgemv(cb,
                    CUBLAS_OP_T, p.dim1, p.dim2, alpha,
                    S, p.s_lda, x, incx,
                    beta, y + y2off, incy));
            }
        } else {
            /* dim1==0 (n=1, UPLO=U): no off-diagonal contribution exists, and
             * SGEMV with the off-diagonal dimension==0 is a BLAS no-op that
             * does NOT apply beta.  Carry beta through to step4's SSYMV.      */
            step4_beta = beta;
        }
    }

    /* Step 4: y[dim1:] = alpha * T2 * x[dim1:] + step4_beta * y[dim1:] */
    if (p.dim2 > 0) {
        CURFP_CHECK_CUBLAS(cublasSsymv(cb,
            p.fill2, p.dim2, alpha,
            arf + p.off2, p.lda2,
            x + (size_t)p.dim1 * incx, incx,
            step4_beta, y + (size_t)p.dim1 * incy, incy));
    }

    return CURFP_STATUS_SUCCESS;
}
