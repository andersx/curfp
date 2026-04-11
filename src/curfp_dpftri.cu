/*
 * curfpDpftri — Compute inverse of an SPD matrix from its RFP Cholesky factor
 *               (double precision).
 *
 * Double-precision copy of curfp_spftri.cu: float -> double,
 * cusolverDnSpotri -> cusolverDnDpotri, cublasStrsm -> cublasDtrsm,
 * cublasSsyrk -> cublasDsyrk, cublasSsymm -> cublasDsymm.
 * float literals -> double literals, sizeof(float) -> sizeof(double).
 * Kernel types: float* -> double*, 1.0f -> 1.0.
 */

#include "curfp_internal.h"

typedef struct {
    cublasFillMode_t  fill11; int dim11; long off11; int lda11;

    cublasSideMode_t  trsm_side;
    cublasFillMode_t  trsm_fill;
    cublasOperation_t trsm_op;
    long              trsm_a_off;
    long              trsm_b_off;
    int               trsm_m, trsm_n;
    int               trsm_lda, trsm_ldb;

    cublasFillMode_t  fill22; int dim22; long off22; int lda22;
} dpftri_params_t;

static void get_dpftri_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               dpftri_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = 0;    p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = 0; p->trsm_b_off = n1; p->trsm_lda = n; p->trsm_ldb = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n;  p->lda22 = n;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = n2; p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = n2; p->trsm_b_off = 0; p->trsm_lda = n; p->trsm_ldb = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n1; p->lda22 = n;
            }
        } else {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = 0; p->lda11 = n1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = 0; p->trsm_b_off = (long)n1*n1; p->trsm_lda = n1; p->trsm_ldb = n1;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = 1; p->lda22 = n1;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = (long)n2*n2;  p->lda11 = n2;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = (long)n2*n2; p->trsm_b_off = 0; p->trsm_lda = n2; p->trsm_ldb = n2;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = (long)n1*n2; p->lda22 = n2;
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = 1;    p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = 1; p->trsm_b_off = nk+1; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = 0;    p->lda22 = n+1;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = nk+1; p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk+1; p->trsm_b_off = 0; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = nk;   p->lda22 = n+1;
            }
        } else {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = nk;           p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk; p->trsm_b_off = (long)nk*(nk+1); p->trsm_lda = nk; p->trsm_ldb = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = 0;            p->lda22 = nk;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = (long)nk*(nk+1); p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = (long)nk*(nk+1); p->trsm_b_off = 0; p->trsm_lda = nk; p->trsm_ldb = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = (long)nk*nk;     p->lda22 = nk;
            }
        }
    }
}

__global__ static void k_inv_sq_d(double *x) { x[0] = 1.0 / (x[0] * x[0]); }

__global__ static void k_copy_mat_d(const double *src, int src_ld,
                                     double *dst, int dst_ld,
                                     int m, int n)
{
    int r = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r < m && c < n) dst[(long)c * dst_ld + r] = src[(long)c * src_ld + r];
}

extern "C"
curfpStatus_t curfpDpftri(curfpHandle_t    handle,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           double          *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    cublasHandle_t     cb = handle->cublas;
    cusolverDnHandle_t cs = handle->cusolver;

    if (n == 1) {
        cudaStream_t stream;
        if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS)
            return CURFP_STATUS_EXECUTION_FAILED;
        k_inv_sq_d<<<1, 1, 0, stream>>>(arf);
        return (cudaGetLastError() == cudaSuccess) ? CURFP_STATUS_SUCCESS
                                                   : CURFP_STATUS_EXECUTION_FAILED;
    }

    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd       = (n % 2 != 0);
    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n/2; n1 = n - n2; }
        else       { n1 = n/2; n2 = n - n1; }
    } else {
        nk = n/2; n1 = nk; n2 = nk;
    }
    dpftri_params_t p;
    get_dpftri_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    int m     = p.trsm_m;
    int n_blk = p.trsm_n;

    cudaStream_t stream;
    if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;

    int lwork11 = 0, lwork22 = 0, lwork;
    {
        cusolverStatus_t s;
        s = cusolverDnDpotri_bufferSize(cs, p.fill11, p.dim11,
                                        arf + p.off11, p.lda11, &lwork11);
        if (s != CUSOLVER_STATUS_SUCCESS) return from_cusolver_status(s);
        s = cusolverDnDpotri_bufferSize(cs, p.fill22, p.dim22,
                                        arf + p.off22, p.lda22, &lwork22);
        if (s != CUSOLVER_STATUS_SUCCESS) return from_cusolver_status(s);
    }
    lwork = (lwork11 > lwork22) ? lwork11 : lwork22;
    if (lwork < 1) lwork = 1;

    double *work    = NULL;
    double *g_buf   = NULL;
    double *h_buf   = NULL;
    int    *devInfo = NULL;

    curfpStatus_t status = CURFP_STATUS_SUCCESS;
    int h_info = 0;
    const double one  =  1.0;
    const double mone = -1.0;
    const double zero =  0.0;

    const cublasOperation_t op_opp = (p.trsm_op == CUBLAS_OP_T) ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int fill22_upper = (p.fill22 == CUBLAS_FILL_MODE_UPPER);
    const int side_left    = (p.trsm_side == CUBLAS_SIDE_LEFT);
    const cublasOperation_t op_h   = (fill22_upper ^ side_left) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasSideMode_t  h_side = side_left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;

#define CHK_CS(expr) \
    do { cusolverStatus_t _s = (expr); \
         if (_s != CUSOLVER_STATUS_SUCCESS) { \
             status = from_cusolver_status(_s); goto cleanup; } \
    } while (0)
#define CHK_CB(expr) \
    do { cublasStatus_t _s = (expr); \
         if (_s != CUBLAS_STATUS_SUCCESS) { \
             status = from_cublas_status(_s); goto cleanup; } \
    } while (0)
#define CHK_CU(expr) \
    do { cudaError_t _e = (expr); \
         if (_e != cudaSuccess) { \
             status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; } \
    } while (0)

    CHK_CU(cudaMallocAsync((void **)&work,    (size_t)lwork     * sizeof(double), stream));
    CHK_CU(cudaMallocAsync((void **)&g_buf,   (size_t)m * n_blk * sizeof(double), stream));
    CHK_CU(cudaMallocAsync((void **)&h_buf,   (size_t)m * n_blk * sizeof(double), stream));
    CHK_CU(cudaMallocAsync((void **)&devInfo, sizeof(int),                         stream));

    {
        dim3 blk(16, 16);
        dim3 grd((n_blk + 15)/16, (m + 15)/16);
        k_copy_mat_d<<<grd, blk, 0, stream>>>(
            arf + p.trsm_b_off, p.trsm_ldb,
            g_buf, m,
            m, n_blk);
        CHK_CU(cudaGetLastError());
    }

    CHK_CB(cublasDtrsm(cb,
        p.trsm_side, p.trsm_fill, op_opp, CUBLAS_DIAG_NON_UNIT,
        m, n_blk, &one,
        arf + p.trsm_a_off, p.trsm_lda,
        g_buf, m));

    {
        dim3 blk(16, 16);
        dim3 grd((n_blk + 15)/16, (m + 15)/16);
        k_copy_mat_d<<<grd, blk, 0, stream>>>(g_buf, m, h_buf, m, m, n_blk);
        CHK_CU(cudaGetLastError());
    }
    CHK_CB(cublasDtrsm(cb,
        h_side, p.fill22, op_h, CUBLAS_DIAG_NON_UNIT,
        m, n_blk, &one,
        arf + p.off22, p.lda22,
        h_buf, m));

    CHK_CU(cudaMemsetAsync(devInfo, 0, sizeof(int), stream));
    CHK_CS(cusolverDnDpotri(cs, p.fill22, p.dim22,
                             arf + p.off22, p.lda22, work, lwork, devInfo));
    CHK_CU(cudaMemcpyAsync(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHK_CU(cudaStreamSynchronize(stream));
    if (h_info != 0) { status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; }

    CHK_CU(cudaMemsetAsync(devInfo, 0, sizeof(int), stream));
    CHK_CS(cusolverDnDpotri(cs, p.fill11, p.dim11,
                             arf + p.off11, p.lda11, work, lwork, devInfo));
    CHK_CU(cudaMemcpyAsync(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHK_CU(cudaStreamSynchronize(stream));
    if (h_info != 0) { status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; }

    if (p.trsm_side == CUBLAS_SIDE_RIGHT) {
        CHK_CB(cublasDsyrk(cb,
            p.fill11, CUBLAS_OP_T,
            n_blk, m,
            &one,  h_buf, m,
            &one,  arf + p.off11, p.lda11));
    } else {
        CHK_CB(cublasDsyrk(cb,
            p.fill11, CUBLAS_OP_N,
            m, n_blk,
            &one,  h_buf, m,
            &one,  arf + p.off11, p.lda11));
    }

    if (p.trsm_side == CUBLAS_SIDE_RIGHT) {
        CHK_CB(cublasDsymm(cb,
            CUBLAS_SIDE_LEFT, p.fill22,
            m, n_blk, &mone,
            arf + p.off22, p.lda22,
            g_buf, m,
            &zero, arf + p.trsm_b_off, p.trsm_ldb));
    } else {
        CHK_CB(cublasDsymm(cb,
            CUBLAS_SIDE_RIGHT, p.fill22,
            m, n_blk, &mone,
            arf + p.off22, p.lda22,
            g_buf, m,
            &zero, arf + p.trsm_b_off, p.trsm_ldb));
    }

#undef CHK_CS
#undef CHK_CB
#undef CHK_CU

cleanup:
    cudaFreeAsync(work,    stream);
    cudaFreeAsync(g_buf,   stream);
    cudaFreeAsync(h_buf,   stream);
    cudaFreeAsync(devInfo, stream);
    return status;
}
