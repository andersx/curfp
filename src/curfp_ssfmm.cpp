/*
 * curfpSsfmm — Symmetric matrix-matrix multiply in RFP format (single prec.)
 *
 * Computes:
 *   side == CURFP_SIDE_LEFT:   C := alpha * A * B + beta * C
 *   side == CURFP_SIDE_RIGHT:  C := alpha * B * A + beta * C
 *
 * where A is an n_A×n_A symmetric matrix in RFP format:
 *   side=LEFT:  n_A = m  (A acts on rows of B and C)
 *   side=RIGHT: n_A = n  (A acts on cols of B and C)
 * B and C are m×n column-major dense matrices.
 *
 * Algorithm: 2×2 block decomposition — NO workspace allocation.
 *
 * Let n_A be the order of A, split at dim1: A = [[T1, ?], [?, T2]] with
 * off-diagonal block S.  For side=LEFT, B and C are split at row dim1:
 *
 *   C1 = alpha*T1*B1 + alpha*S_mv*B2 + beta*C1   → ssymm(T1) + sgemm(S)
 *   C2 = alpha*S_mv'*B1 + alpha*T2*B2 + beta*C2  → sgemm(S)  + ssymm(T2)
 *
 * For side=RIGHT, B and C are split at column dim1:
 *
 *   C[:,0:d1] = alpha*B1*T1 + alpha*B2*S_mv' + beta*C[:,0:d1]
 *   C[:,d1:]  = alpha*B1*S_mv + alpha*B2*T2  + beta*C[:,d1:]
 *
 * The sgemm op codes are derived from s_op1 in the sub-block params table,
 * exactly mirroring the ssfmv steps 2&3 (sgemv → sgemm with extra n_rhs dim).
 *
 * Column-major convention: B and C have ldb/ldc as leading dimension.
 * Sub-block row offsets → pointer += dim1 (side=LEFT).
 * Sub-block col offsets → pointer += dim1 * ldb/ldc (side=RIGHT).
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Sub-block layout — identical to curfp_ssfmv.cpp / curfp_ssfr.cpp.
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
} ssfmm_params_t;

static void get_ssfmm_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               ssfmm_params_t *p)
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
curfpStatus_t curfpSsfmm(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpSideMode_t  side,
    int              m,
    int              n,
    const float     *alpha,
    const float     *arf,
    const float     *B,
    int              ldb,
    const float     *beta,
    float           *C,
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

    /* n_A: order of the symmetric matrix A */
    const int n_A   = left ? m : n;
    const int nisodd = (n_A % 2 != 0);

    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n_A / 2; n1 = n_A - n2; }
        else       { n1 = n_A / 2; n2 = n_A - n1; }
    } else {
        nk = n_A / 2; n1 = nk; n2 = nk;
    }

    ssfmm_params_t p;
    get_ssfmm_params(nisodd, normaltransr, lower, n_A, n1, n2, nk, &p);

    const float  one = 1.0f;
    const float *S   = arf + p.s_off;

    if (left) {
        /* -------------------------------------------------------------------
         * side=LEFT:  C(m×n) = alpha * A(m×m) * B(m×n) + beta * C(m×n)
         *
         * Row-split B and C at row dim1.
         *   B1 = B[0:dim1, :],   B2 = B[dim1:, :]
         *   C1 = C[0:dim1, :],   C2 = C[dim1:, :]
         *
         * Column-major: B1 = B (ptr), B2 = B + dim1 (pointer offset by dim1 rows).
         * Similarly for C.
         * ------------------------------------------------------------------- */
        const float *B2 = B + p.dim1;
        float       *C2 = C + p.dim1;

        /* Step 1: C1 = alpha * T1 * B1 + beta * C1 */
        CURFP_CHECK_CUBLAS(cublasSsymm(cb,
            CUBLAS_SIDE_LEFT, p.fill1,
            p.dim1, n,
            alpha, arf + p.off1, p.lda1,
            B, ldb,
            beta, C, ldc));

        if (p.dim2 > 0) {
            /* Step 2: C1 += alpha * S_part * B2
             *
             * s_op1 == TRANS:   S is dim2×dim1  → need S^T (dim1×dim2) * B2(dim2×n)
             *   sgemm(TRANS, NOTRANS, dim1, n, dim2, alpha, S, s_lda, B2, ldb, 1, C1, ldc)
             *
             * s_op1 == NOTRANS: S is dim1×dim2  → need S(dim1×dim2) * B2(dim2×n)
             *   sgemm(NOTRANS, NOTRANS, dim1, n, dim2, alpha, S, s_lda, B2, ldb, 1, C1, ldc)
             */
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    p.dim1, n, p.dim2,
                    alpha, S, p.s_lda, B2, ldb,
                    &one, C, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    p.dim1, n, p.dim2,
                    alpha, S, p.s_lda, B2, ldb,
                    &one, C, ldc));
            }

            /* Step 3: C2 = alpha * S_part' * B1 + beta * C2
             *
             * s_op1 == TRANS:   S is dim2×dim1  → S(dim2×dim1) * B1(dim1×n)
             *   sgemm(NOTRANS, NOTRANS, dim2, n, dim1, alpha, S, s_lda, B, ldb, beta, C2, ldc)
             *
             * s_op1 == NOTRANS: S is dim1×dim2  → S^T(dim2×dim1) * B1(dim1×n)
             *   sgemm(TRANS, NOTRANS, dim2, n, dim1, alpha, S, s_lda, B, ldb, beta, C2, ldc)
             */
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    p.dim2, n, p.dim1,
                    alpha, S, p.s_lda, B, ldb,
                    beta, C2, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    p.dim2, n, p.dim1,
                    alpha, S, p.s_lda, B, ldb,
                    beta, C2, ldc));
            }

            /* Step 4: C2 += alpha * T2 * B2 */
            CURFP_CHECK_CUBLAS(cublasSsymm(cb,
                CUBLAS_SIDE_LEFT, p.fill2,
                p.dim2, n,
                alpha, arf + p.off2, p.lda2,
                B2, ldb,
                &one, C2, ldc));
        }

    } else {
        /* -------------------------------------------------------------------
         * side=RIGHT:  C(m×n) = alpha * B(m×n) * A(n×n) + beta * C(m×n)
         *
         * Column-split B and C at column dim1.
         *   B1 = B[:, 0:dim1],   B2 = B[:, dim1:]
         *   C1 = C[:, 0:dim1],   C2 = C[:, dim1:]
         *
         * Column-major: B1 = B (ptr), B2 = B + dim1*ldb.
         * Similarly for C.
         * ------------------------------------------------------------------- */
        const float *B2 = B + (size_t)p.dim1 * ldb;
        float       *C2 = C + (size_t)p.dim1 * ldc;

        /* Step 1: C1 = alpha * B1 * T1 + beta * C1 */
        CURFP_CHECK_CUBLAS(cublasSsymm(cb,
            CUBLAS_SIDE_RIGHT, p.fill1,
            m, p.dim1,
            alpha, arf + p.off1, p.lda1,
            B, ldb,
            beta, C, ldc));

        if (p.dim2 > 0) {
            /* Step 2: C1 += alpha * B2 * S_part
             *
             * s_op1 == TRANS:   S is dim2×dim1  → B2(m×dim2) * S(dim2×dim1)
             *   sgemm(NOTRANS, NOTRANS, m, dim1, dim2, alpha, B2, ldb, S, s_lda, 1, C1, ldc)
             *
             * s_op1 == NOTRANS: S is dim1×dim2  → B2(m×dim2) * S^T(dim2×dim1)
             *   sgemm(NOTRANS, TRANS, m, dim1, dim2, alpha, B2, ldb, S, s_lda, 1, C1, ldc)
             */
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, p.dim1, p.dim2,
                    alpha, B2, ldb, S, p.s_lda,
                    &one, C, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, p.dim1, p.dim2,
                    alpha, B2, ldb, S, p.s_lda,
                    &one, C, ldc));
            }

            /* Step 3: C2 = alpha * B1 * S_part' + beta * C2
             *
             * s_op1 == TRANS:   S is dim2×dim1  → B1(m×dim1) * S^T(dim1×dim2)
             *   sgemm(NOTRANS, TRANS, m, dim2, dim1, alpha, B, ldb, S, s_lda, beta, C2, ldc)
             *
             * s_op1 == NOTRANS: S is dim1×dim2  → B1(m×dim1) * S(dim1×dim2)
             *   sgemm(NOTRANS, NOTRANS, m, dim2, dim1, alpha, B, ldb, S, s_lda, beta, C2, ldc)
             */
            if (p.s_op1 == CUBLAS_OP_T) {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, p.dim2, p.dim1,
                    alpha, B, ldb, S, p.s_lda,
                    beta, C2, ldc));
            } else {
                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, p.dim2, p.dim1,
                    alpha, B, ldb, S, p.s_lda,
                    beta, C2, ldc));
            }

            /* Step 4: C2 += alpha * B2 * T2 */
            CURFP_CHECK_CUBLAS(cublasSsymm(cb,
                CUBLAS_SIDE_RIGHT, p.fill2,
                m, p.dim2,
                alpha, arf + p.off2, p.lda2,
                B2, ldb,
                &one, C2, ldc));
        }
    }

    return CURFP_STATUS_SUCCESS;
}
