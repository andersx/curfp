/*
 * curfpDpftrf — Cholesky factorization in RFP format (double precision)
 *
 * Double-precision copy of curfp_spftrf.cpp: float -> double,
 * cusolverDnSpotrf -> cusolverDnDpotrf, cublasStrsm -> cublasDtrsm,
 * cublasSsyrk -> cublasDsyrk.
 */

#include <stdlib.h>
#include "curfp_internal.h"

typedef struct {
    cublasFillMode_t  fill11;
    int               dim11;
    long              off11;
    int               lda11;

    cublasSideMode_t  trsm_side;
    cublasFillMode_t  trsm_fill;
    cublasOperation_t trsm_op;
    long              trsm_a_off;
    long              trsm_b_off;
    int               trsm_m, trsm_n;
    int               trsm_lda, trsm_ldb;

    cublasFillMode_t  syrk_fill;
    cublasOperation_t syrk_op;
    int               syrk_n, syrk_k;
    long              syrk_a_off;
    int               syrk_lda;
    long              syrk_c_off;
    int               syrk_ldc;

    cublasFillMode_t  fill22;
    int               dim22;
    long              off22;
    int               lda22;
    int               info22_offset;
} dpftrf_params_t;

static void get_dpftrf_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               dpftrf_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = 0;    p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = 0; p->trsm_b_off = n1; p->trsm_lda = n; p->trsm_ldb = n;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = n1; p->syrk_lda = n;
                p->syrk_c_off = n; p->syrk_ldc = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n;  p->lda22 = n;
                p->info22_offset = n1;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = n2;  p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = n2; p->trsm_b_off = 0; p->trsm_lda = n; p->trsm_ldb = n;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = 0; p->syrk_lda = n;
                p->syrk_c_off = n1; p->syrk_ldc = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n1; p->lda22 = n;
                p->info22_offset = n1;
            }
        } else {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = 0; p->lda11 = n1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = 0; p->trsm_b_off = (long)n1*n1; p->trsm_lda = n1; p->trsm_ldb = n1;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = (long)n1*n1; p->syrk_lda = n1;
                p->syrk_c_off = 1; p->syrk_ldc = n1;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = 1; p->lda22 = n1;
                p->info22_offset = n1;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = (long)n2*n2; p->lda11 = n2;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = (long)n2*n2; p->trsm_b_off = 0; p->trsm_lda = n2; p->trsm_ldb = n2;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = n2; p->syrk_k = n1; p->syrk_a_off = 0; p->syrk_lda = n2;
                p->syrk_c_off = (long)n1*n2; p->syrk_ldc = n2;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = (long)n1*n2; p->lda22 = n2;
                p->info22_offset = n1;
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = 1;    p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = 1; p->trsm_b_off = nk+1; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = nk+1; p->syrk_lda = n+1;
                p->syrk_c_off = 0; p->syrk_ldc = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = 0;    p->lda22 = n+1;
                p->info22_offset = nk;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = nk+1; p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk+1; p->trsm_b_off = 0; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->syrk_fill = CUBLAS_FILL_MODE_UPPER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = 0; p->syrk_lda = n+1;
                p->syrk_c_off = nk; p->syrk_ldc = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = nk;   p->lda22 = n+1;
                p->info22_offset = nk;
            }
        } else {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = nk;           p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk; p->trsm_b_off = (long)nk*(nk+1); p->trsm_lda = nk; p->trsm_ldb = nk;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_T;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = (long)nk*(nk+1); p->syrk_lda = nk;
                p->syrk_c_off = 0; p->syrk_ldc = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = 0;            p->lda22 = nk;
                p->info22_offset = nk;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = (long)nk*(nk+1); p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = (long)nk*(nk+1); p->trsm_b_off = 0; p->trsm_lda = nk; p->trsm_ldb = nk;
                p->syrk_fill = CUBLAS_FILL_MODE_LOWER; p->syrk_op = CUBLAS_OP_N;
                p->syrk_n = nk; p->syrk_k = nk; p->syrk_a_off = 0; p->syrk_lda = nk;
                p->syrk_c_off = (long)nk*nk; p->syrk_ldc = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = (long)nk*nk; p->lda22 = nk;
                p->info22_offset = nk;
            }
        }
    }
}

curfpStatus_t curfpDpftrf(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    double          *A,
    int             *info)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0)  return CURFP_STATUS_INVALID_VALUE;
    if (!info)  return CURFP_STATUS_INVALID_VALUE;
    *info = 0;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    cublasHandle_t     cb = handle->cublas;
    cusolverDnHandle_t cs = handle->cusolver;

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

    dpftrf_params_t p;
    get_dpftrf_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const double one  =  1.0;
    const double mone = -1.0;

    int lwork11 = 0, lwork22 = 0;
    CURFP_CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cs, p.fill11, p.dim11,
                                                      A + p.off11, p.lda11, &lwork11));
    CURFP_CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cs, p.fill22, p.dim22,
                                                      A + p.off22, p.lda22, &lwork22));
    int lwork = (lwork11 > lwork22) ? lwork11 : lwork22;
    if (lwork < 1) lwork = 1;

    cudaStream_t stream;
    CURFP_CHECK_CUBLAS(cublasGetStream(cb, &stream));

    double *work    = NULL;
    int    *devInfo = NULL;
    CURFP_CHECK_CUDA(cudaMallocAsync((void **)&work,    (size_t)lwork * sizeof(double), stream));
    CURFP_CHECK_CUDA(cudaMallocAsync((void **)&devInfo, sizeof(int),                    stream));

    curfpStatus_t st    = CURFP_STATUS_SUCCESS;
    int           h_info = 0;

#define CHK_CS(expr) \
    do { cusolverStatus_t _s = (expr); \
         if (_s != CUSOLVER_STATUS_SUCCESS) { \
             st = from_cusolver_status(_s); goto cleanup; } \
    } while (0)
#define CHK_CB(expr) \
    do { cublasStatus_t _s = (expr); \
         if (_s != CUBLAS_STATUS_SUCCESS) { \
             st = from_cublas_status(_s); goto cleanup; } \
    } while (0)
#define CHK_CU(expr) \
    do { cudaError_t _e = (expr); \
         if (_e != cudaSuccess) { \
             st = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; } \
    } while (0)

    CHK_CU(cudaMemsetAsync(devInfo, 0, sizeof(int), stream));
    CHK_CS(cusolverDnDpotrf(cs, p.fill11, p.dim11,
                             A + p.off11, p.lda11, work, lwork, devInfo));
    CHK_CU(cudaMemcpyAsync(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHK_CU(cudaStreamSynchronize(stream));
    if (h_info != 0) { *info = h_info; goto cleanup; }

    CHK_CB(cublasDtrsm(cb,
        p.trsm_side, p.trsm_fill, p.trsm_op, CUBLAS_DIAG_NON_UNIT,
        p.trsm_m, p.trsm_n, &one,
        A + p.trsm_a_off, p.trsm_lda,
        A + p.trsm_b_off, p.trsm_ldb));

    CHK_CB(cublasDsyrk(cb,
        p.syrk_fill, p.syrk_op,
        p.syrk_n, p.syrk_k, &mone,
        A + p.syrk_a_off, p.syrk_lda,
        &one, A + p.syrk_c_off, p.syrk_ldc));

    CHK_CU(cudaMemsetAsync(devInfo, 0, sizeof(int), stream));
    CHK_CS(cusolverDnDpotrf(cs, p.fill22, p.dim22,
                             A + p.off22, p.lda22, work, lwork, devInfo));
    CHK_CU(cudaMemcpyAsync(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHK_CU(cudaStreamSynchronize(stream));
    if (h_info != 0) { *info = h_info + p.info22_offset; goto cleanup; }

#undef CHK_CS
#undef CHK_CB
#undef CHK_CU

cleanup:
    cudaFreeAsync(work,    stream);
    cudaFreeAsync(devInfo, stream);
    return st;
}
