/*
 * curfpDlansf — Norm of a symmetric matrix in RFP format (double precision)
 *
 * Double-precision copy of curfp_slansf.cu: float -> double,
 * fabsf -> fabs, sqrtf -> sqrt,
 * cublasIsamax -> cublasIdamax, cublasSnrm2 -> cublasDnrm2.
 */

#include <math.h>
#include <climits>
#include "curfp_internal.h"

static const int BLOCK = 256;

typedef struct {
    cublasFillMode_t  fill1; int dim1; long off1; int lda1;
    cublasFillMode_t  fill2; int dim2; long off2; int lda2;
    long              s_off; int s_lda; cublasOperation_t s_op1;
} dlansf_params_t;

static void get_dlansf_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               dlansf_params_t *p)
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

static __global__ void k_colsum_sym_d(
    const double * __restrict__ arf,
    long off, int dim, int lda, int fill_upper,
    int col_base,
    double * __restrict__ work)
{
    int j   = blockIdx.x;
    int tid = threadIdx.x;
    if (j >= dim) return;

    extern __shared__ double sdata_d[];

    const double *T = arf + off;
    double s = 0.0;

    if (fill_upper) {
        for (int r = tid; r < dim; r += blockDim.x) {
            double v;
            if (r <= j)
                v = T[(long)j * lda + r];
            else
                v = T[(long)r * lda + j];
            s += fabs(v);
        }
    } else {
        for (int r = tid; r < dim; r += blockDim.x) {
            double v;
            if (r >= j)
                v = T[(long)j * lda + r];
            else
                v = T[(long)r * lda + j];
            s += fabs(v);
        }
    }

    sdata_d[tid] = s;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata_d[tid] += sdata_d[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        work[col_base + j] = sdata_d[0];
}

static __global__ void k_colsum_rect_d(
    const double * __restrict__ S,
    int rows, int cols, int lda,
    int is_a21, int dim1,
    double * __restrict__ work)
{
    int j   = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ double sdata_d[];

    double s = 0.0;

    if (is_a21) {
        if (j < cols) {
            for (int r = tid; r < rows; r += blockDim.x)
                s += fabs(S[(long)j * lda + r]);
            sdata_d[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata_d[tid] += sdata_d[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[j], sdata_d[0]);
            __syncthreads();
        }

        s = 0.0;
        if (j < rows) {
            for (int c = tid; c < cols; c += blockDim.x)
                s += fabs(S[(long)c * lda + j]);
            sdata_d[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata_d[tid] += sdata_d[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[dim1 + j], sdata_d[0]);
        }
    } else {
        if (j < rows) {
            for (int c = tid; c < cols; c += blockDim.x)
                s += fabs(S[(long)c * lda + j]);
            sdata_d[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata_d[tid] += sdata_d[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[j], sdata_d[0]);
            __syncthreads();
        }

        s = 0.0;
        if (j < cols) {
            for (int r = tid; r < rows; r += blockDim.x)
                s += fabs(S[(long)j * lda + r]);
            sdata_d[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata_d[tid] += sdata_d[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[dim1 + j], sdata_d[0]);
        }
    }
}

extern "C"
curfpStatus_t curfpDlansf(curfpHandle_t    handle,
                           curfpNormType_t  norm,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           const double    *arf,
                           double          *result)
{
    CURFP_CHECK_HANDLE(handle);
    if (!result) return CURFP_STATUS_INVALID_VALUE;
    if (n < 0)   return CURFP_STATUS_INVALID_VALUE;

    *result = 0.0;
    if (n == 0)  return CURFP_STATUS_SUCCESS;

    cublasHandle_t cb = handle->cublas;

    cublasPointerMode_t old_mode;
    if (cublasGetPointerMode(cb, &old_mode) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;
    if (cublasSetPointerMode(cb, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;

    curfpStatus_t status = CURFP_STATUS_SUCCESS;
    double       *d_work = NULL;
    long long nt_ll = (long long)n * (n + 1) / 2;
    int       nt    = (nt_ll <= INT_MAX) ? (int)nt_ll : -1;

    cudaStream_t stream;
    if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;

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
    dlansf_params_t p;
    get_dlansf_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

#define CHK_CB(expr) \
    do { cublasStatus_t _cs = (expr); \
         if (_cs != CUBLAS_STATUS_SUCCESS) { \
             status = from_cublas_status(_cs); goto cleanup; } \
    } while (0)
#define CHK_CU(expr) \
    do { cudaError_t _e = (expr); \
         if (_e != cudaSuccess) { \
             status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; } \
    } while (0)

    if (norm == CURFP_NORM_MAX) {
        if (nt < 0) { status = CURFP_STATUS_INVALID_VALUE; goto cleanup; }
        int idx = 0;
        CHK_CB(cublasIdamax(cb, nt, arf, 1, &idx));
        double val = 0.0;
        CHK_CU(cudaMemcpyAsync(&val, arf + (idx - 1), sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
        CHK_CU(cudaStreamSynchronize(stream));
        *result = fabs(val);

    } else if (norm == CURFP_NORM_ONE) {
        CHK_CU(cudaMallocAsync((void **)&d_work, (size_t)n * sizeof(double), stream));

        size_t smem = BLOCK * sizeof(double);
        int upper1 = (p.fill1 == CUBLAS_FILL_MODE_UPPER) ? 1 : 0;
        int upper2 = (p.fill2 == CUBLAS_FILL_MODE_UPPER) ? 1 : 0;

        if (p.dim1 > 0) {
            k_colsum_sym_d<<<p.dim1, BLOCK, smem, stream>>>(
                arf, p.off1, p.dim1, p.lda1, upper1, 0, d_work);
            CHK_CU(cudaGetLastError());
        }
        if (p.dim2 > 0) {
            k_colsum_sym_d<<<p.dim2, BLOCK, smem, stream>>>(
                arf, p.off2, p.dim2, p.lda2, upper2, p.dim1, d_work);
            CHK_CU(cudaGetLastError());
        }

        if (p.dim1 > 0 && p.dim2 > 0) {
            int is_a21 = (p.s_op1 == CUBLAS_OP_T) ? 1 : 0;
            int rows   = is_a21 ? p.dim2 : p.dim1;
            int cols   = is_a21 ? p.dim1 : p.dim2;
            int grid   = max(rows, cols);
            k_colsum_rect_d<<<grid, BLOCK, smem, stream>>>(
                arf + p.s_off, rows, cols, p.s_lda, is_a21, p.dim1, d_work);
            CHK_CU(cudaGetLastError());
        }

        int idx = 0;
        CHK_CB(cublasIdamax(cb, n, d_work, 1, &idx));
        double val = 0.0;
        CHK_CU(cudaMemcpyAsync(&val, d_work + (idx - 1), sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
        CHK_CU(cudaStreamSynchronize(stream));
        *result = val;

    } else if (norm == CURFP_NORM_FRO) {
        if (nt < 0) { status = CURFP_STATUS_INVALID_VALUE; goto cleanup; }

        double arf_nrm   = 0.0;
        double diag1_nrm = 0.0;
        double diag2_nrm = 0.0;

        CHK_CB(cublasDnrm2(cb, nt, arf, 1, &arf_nrm));

        if (p.dim1 > 0)
            CHK_CB(cublasDnrm2(cb, p.dim1, arf + p.off1, p.lda1 + 1, &diag1_nrm));

        if (p.dim2 > 0)
            CHK_CB(cublasDnrm2(cb, p.dim2, arf + p.off2, p.lda2 + 1, &diag2_nrm));

        double fro_sq = 2.0 * arf_nrm  * arf_nrm
                      -       diag1_nrm * diag1_nrm
                      -       diag2_nrm * diag2_nrm;
        *result = (fro_sq > 0.0) ? sqrt(fro_sq) : 0.0;

    } else {
        status = CURFP_STATUS_INVALID_VALUE;
    }

#undef CHK_CB
#undef CHK_CU

cleanup:
    cublasSetPointerMode(cb, old_mode);
    if (d_work) cudaFreeAsync(d_work, stream);
    return status;
}
