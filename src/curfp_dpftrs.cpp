/*
 * curfpDpftrs — Triangular solve using RFP Cholesky factor (double precision)
 *
 * Double-precision copy of curfp_spftrs.cpp: float -> double,
 * cublasStrsm -> cublasDtrsm, cublasSgemm -> cublasDgemm.
 */

#include "curfp_internal.h"

typedef struct {
    int  dim_fwd, dim_bwd;

    cublasFillMode_t  t16_fill;
    long              t16_a_off;
    int               t16_lda;
    cublasOperation_t t1_op;
    cublasOperation_t t6_op;
    long              t16_b_off;

    cublasFillMode_t  t34_fill;
    long              t34_a_off;
    int               t34_lda;
    cublasOperation_t t3_op;
    cublasOperation_t t4_op;
    long              t34_b_off;

    cublasOperation_t g2_op;
    long              g2_a_off;
    int               g2_lda;
    cublasOperation_t g5_op;
    long              g5_a_off;
} dpftrs_params_t;

static void get_dpftrs_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               dpftrs_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = 0;   p->t16_lda = n;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = n;   p->t34_lda = n;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = n1; p->g2_lda = n;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = n1;
            } else {
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = n2;  p->t16_lda = n;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = n1;  p->t34_lda = n;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = 0; p->g2_lda = n;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = 0;
            }
        } else {
            if (lower) {
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = 0;          p->t16_lda = n1;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = 1;          p->t34_lda = n1;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = (long)n1*n1; p->g2_lda = n1;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = (long)n1*n1;
            } else {
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = (long)n2*n2; p->t16_lda = n2;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = (long)n1*n2; p->t34_lda = n2;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = 0; p->g2_lda = n2;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = 0;
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = 1;    p->t16_lda = n+1;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = 0;    p->t34_lda = n+1;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = nk+1; p->g2_lda = n+1;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = nk+1;
            } else {
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = nk+1; p->t16_lda = n+1;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = nk;   p->t34_lda = n+1;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = 0; p->g2_lda = n+1;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = 0;
            }
        } else {
            if (lower) {
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = nk;           p->t16_lda = nk;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = 0;            p->t34_lda = nk;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = (long)nk*(nk+1); p->g2_lda = nk;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = (long)nk*(nk+1);
            } else {
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = (long)nk*(nk+1); p->t16_lda = nk;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = (long)nk*nk;    p->t34_lda = nk;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = 0; p->g2_lda = nk;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = 0;
            }
        }
    }
}

curfpStatus_t curfpDpftrs(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    int              nrhs,
    const double    *A,
    double          *B,
    int              ldb)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || nrhs < 0 || ldb < 1) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0 || nrhs == 0)          return CURFP_STATUS_SUCCESS;

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

    dpftrs_params_t p;
    get_dpftrs_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const double one  =  1.0;
    const double mone = -1.0;

    double *Bfwd = B + p.t16_b_off;
    double *Bbwd = B + p.t34_b_off;
    double *Bg2  = B + p.t34_b_off;
    double *Bg5  = B + p.t16_b_off;

    CURFP_CHECK_CUBLAS(cublasDtrsm(cb,
        CUBLAS_SIDE_LEFT, p.t16_fill, p.t1_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_fwd, nrhs, &one,
        A + p.t16_a_off, p.t16_lda,
        Bfwd, ldb));

    CURFP_CHECK_CUBLAS(cublasDgemm(cb,
        p.g2_op, CUBLAS_OP_N,
        p.dim_bwd, nrhs, p.dim_fwd,
        &mone, A + p.g2_a_off, p.g2_lda,
        Bfwd, ldb,
        &one, Bg2, ldb));

    CURFP_CHECK_CUBLAS(cublasDtrsm(cb,
        CUBLAS_SIDE_LEFT, p.t34_fill, p.t3_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_bwd, nrhs, &one,
        A + p.t34_a_off, p.t34_lda,
        Bbwd, ldb));

    CURFP_CHECK_CUBLAS(cublasDtrsm(cb,
        CUBLAS_SIDE_LEFT, p.t34_fill, p.t4_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_bwd, nrhs, &one,
        A + p.t34_a_off, p.t34_lda,
        Bbwd, ldb));

    CURFP_CHECK_CUBLAS(cublasDgemm(cb,
        p.g5_op, CUBLAS_OP_N,
        p.dim_fwd, nrhs, p.dim_bwd,
        &mone, A + p.g5_a_off, p.g2_lda,
        Bbwd, ldb,
        &one, Bg5, ldb));

    CURFP_CHECK_CUBLAS(cublasDtrsm(cb,
        CUBLAS_SIDE_LEFT, p.t16_fill, p.t6_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_fwd, nrhs, &one,
        A + p.t16_a_off, p.t16_lda,
        Bfwd, ldb));

    return CURFP_STATUS_SUCCESS;
}
