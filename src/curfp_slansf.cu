/*
 * curfpSlansf — Norm of a symmetric matrix in RFP format (single precision)
 *
 * Computes one of:
 *   CURFP_NORM_MAX: max(|A[i,j]|)          — max element absolute value
 *   CURFP_NORM_ONE: max_j sum_i |A[i,j]|   — 1-norm = inf-norm (symmetric)
 *   CURFP_NORM_FRO: sqrt(sum_ij |A[i,j]|²) — Frobenius norm
 *
 * Uses the same 2×2 block decomposition as curfpSsfmv:
 *   A = [[T1, S^T], [S, T2]]
 *
 * 1-norm:
 *   For each column j of the full n×n symmetric A, accumulate the absolute
 *   column sum using a 2D block reduction: one CUDA block per column,
 *   threads cooperatively reduce over all n rows via shared memory.
 *   Then take the max over work[0..n-1].
 *
 * Frobenius norm:
 *   ||A||_F² = 2·||arf||² − ||diag(T1)||² − ||diag(T2)||²
 *   Uses cublasSnrm2 (not cublasSdot) to avoid float32 accumulation overflow
 *   at large n.  Each off-diagonal element appears once in arf but contributes
 *   twice to ||A||_F² (A is symmetric, so A[i,j]=A[j,i]).
 *
 * Max-element norm:
 *   cublasIsamax on the full arf array (every stored element is a unique A[i,j]).
 */

#include <math.h>
#include <climits>
#include "curfp_internal.h"

/* Threads per block for 1-norm reduction (must be power of 2) */
static const int BLOCK = 256;

/* -------------------------------------------------------------------------
 * Per-case parameters (identical to ssfmv_params_t / get_ssfmv_params in
 * curfp_ssfmv.cpp — duplicated here so slansf has no link-time dependency
 * on the ssfmv translation unit).
 * ------------------------------------------------------------------------- */
typedef struct {
    cublasFillMode_t  fill1; int dim1; long off1; int lda1;
    cublasFillMode_t  fill2; int dim2; long off2; int lda2;
    long              s_off; int s_lda; cublasOperation_t s_op1;
} slansf_params_t;

static void get_slansf_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               slansf_params_t *p)
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

/* =========================================================================
 * 1-norm: 2D block reduction kernels
 *
 * Strategy: one CUDA block per output column of the full n×n symmetric A.
 * Threads within the block cooperatively reduce over all n rows using
 * shared memory, then thread 0 writes the column sum to work[col].
 *
 * This gives O(n/BLOCK) work per thread (vs O(n) in the old 1D kernels)
 * and launches n blocks for full GPU occupancy.
 *
 * Two kernels:
 *   k_colsum_sym  — handles T1 and T2 triangular sub-blocks (initialises work)
 *   k_colsum_rect — adds S rectangular sub-block contributions (atomicAdd)
 *
 * The atomic in k_colsum_rect is fine: each column of work[] is written by
 * exactly one k_colsum_sym block (which finishes first) and one
 * k_colsum_rect block — only 2 writers total per slot.
 * ========================================================================= */

/*
 * k_colsum_sym: column absolute sums for one triangular sub-block T.
 *
 * T is (dim × dim), col-major with leading dimension lda, base pointer
 * arf+off.  fill_upper=1 means the upper triangle is stored.
 *
 * Each block handles one column j (blockIdx.x).
 * Threads stride over rows in steps of blockDim.x, summing |T[r,j]| for
 * stored elements and |T[j,r]| for reflected elements (symmetric).
 * Shared-memory tree reduction gives the final sum.
 * Result written to work[col_global] where col_global = base + j.
 */
static __global__ void k_colsum_sym(
    const float * __restrict__ arf,
    long off, int dim, int lda, int fill_upper,
    int col_base,
    float * __restrict__ work)
{
    int j   = blockIdx.x;   /* column within this sub-block */
    int tid = threadIdx.x;
    if (j >= dim) return;

    extern __shared__ float sdata[];

    const float *T = arf + off;
    float s = 0.0f;

    if (fill_upper) {
        /* stored: rows 0..j (column j of upper triangular)
         * reflected: rows j+1..dim-1 (read from row j of columns j+1..dim-1) */
        for (int r = tid; r < dim; r += blockDim.x) {
            float v;
            if (r <= j)
                v = T[(long)j * lda + r];   /* stored element */
            else
                v = T[(long)r * lda + j];   /* reflected: T[j,r] stored at col r */
            s += fabsf(v);
        }
    } else {
        /* stored: rows j..dim-1 (column j of lower triangular)
         * reflected: rows 0..j-1 */
        for (int r = tid; r < dim; r += blockDim.x) {
            float v;
            if (r >= j)
                v = T[(long)j * lda + r];   /* stored element */
            else
                v = T[(long)r * lda + j];   /* reflected */
            s += fabsf(v);
        }
    }

    sdata[tid] = s;
    __syncthreads();

    /* Standard power-of-2 tree reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        work[col_base + j] = sdata[0];
}

/*
 * k_colsum_rect: add S rectangular sub-block contributions to work[].
 *
 * S is column-major, shape (rows × cols), pointer arf+s_off, lda=s_lda.
 * is_a21=1: S = A[dim1:n, 0:dim1]  (rows=dim2, cols=dim1)
 *   col j < dim1: sum of |S[r,j]| for r=0..dim2-1 → atomicAdd(work[j], sum)
 *   row j < dim2: sum of |S[j,c]| for c=0..dim1-1 → atomicAdd(work[dim1+j], sum)
 * is_a21=0: S = A[0:dim1, dim1:n]  (rows=dim1, cols=dim2)
 *   row j < dim1: → atomicAdd(work[j], sum)
 *   col j < dim2: → atomicAdd(work[dim1+j], sum)
 *
 * One block per column-of-interest; threads stride over the other dimension.
 */
static __global__ void k_colsum_rect(
    const float * __restrict__ S,
    int rows, int cols, int lda,
    int is_a21, int dim1,
    float * __restrict__ work)
{
    int j   = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    float s = 0.0f;

    if (is_a21) {
        /* Two roles for j: as a column index (j<cols=dim1) and as a row index (j<rows=dim2).
         * Launch with gridDim.x = max(cols, rows); each block handles both roles. */
        if (j < cols) {
            /* column j of S → work[j] */
            for (int r = tid; r < rows; r += blockDim.x)
                s += fabsf(S[(long)j * lda + r]);
            sdata[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata[tid] += sdata[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[j], sdata[0]);
            __syncthreads();
        }

        s = 0.0f;
        if (j < rows) {
            /* row j of S → work[dim1+j] */
            for (int c = tid; c < cols; c += blockDim.x)
                s += fabsf(S[(long)c * lda + j]);
            sdata[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata[tid] += sdata[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[dim1 + j], sdata[0]);
        }
    } else {
        /* is_a21=0: S = A[0:dim1, dim1:n], rows=dim1, cols=dim2 */
        if (j < rows) {
            /* row j of S → work[j] */
            for (int c = tid; c < cols; c += blockDim.x)
                s += fabsf(S[(long)c * lda + j]);
            sdata[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata[tid] += sdata[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[j], sdata[0]);
            __syncthreads();
        }

        s = 0.0f;
        if (j < cols) {
            /* column j of S → work[dim1+j] */
            for (int r = tid; r < rows; r += blockDim.x)
                s += fabsf(S[(long)j * lda + r]);
            sdata[tid] = s;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) sdata[tid] += sdata[tid + stride];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(&work[dim1 + j], sdata[0]);
        }
    }
}

/* =========================================================================
 * Public entry point
 * ========================================================================= */
extern "C"
curfpStatus_t curfpSlansf(curfpHandle_t    handle,
                           curfpNormType_t  norm,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           const float     *arf,
                           float           *result)
{
    CURFP_CHECK_HANDLE(handle);
    if (!result) return CURFP_STATUS_INVALID_VALUE;
    if (n < 0)   return CURFP_STATUS_INVALID_VALUE;

    *result = 0.0f;
    if (n == 0)  return CURFP_STATUS_SUCCESS;

    cublasHandle_t cb = handle->cublas;

    /* Save and set pointer mode to HOST so cuBLAS scalars come back to host */
    cublasPointerMode_t old_mode;
    if (cublasGetPointerMode(cb, &old_mode) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;
    if (cublasSetPointerMode(cb, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;

    curfpStatus_t status = CURFP_STATUS_SUCCESS;
    float        *d_work = NULL;
    /* n*(n+1)/2 — use long long to avoid int overflow for large n.
     * cublasSnrm2 / cublasIsamax take int n, so cap at INT_MAX; -1 means "too large". */
    long long nt_ll = (long long)n * (n + 1) / 2;
    int       nt    = (nt_ll <= INT_MAX) ? (int)nt_ll : -1;

    cudaStream_t stream;
    if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;

    /* Compute block decomposition parameters */
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
    slansf_params_t p;
    get_slansf_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

/* Local error-handling macros (goto cleanup, restore pointer mode) */
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
        /* max(|A[i,j]|) — every element of arf is a unique A entry */
        if (nt < 0) { status = CURFP_STATUS_INVALID_VALUE; goto cleanup; }
        int idx = 0;
        CHK_CB(cublasIsamax(cb, nt, arf, 1, &idx));
        float val = 0.0f;
        CHK_CU(cudaMemcpyAsync(&val, arf + (idx - 1), sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
        CHK_CU(cudaStreamSynchronize(stream));
        *result = fabsf(val);

    } else if (norm == CURFP_NORM_ONE) {
        /*
         * max column absolute sum.
         *
         * k_colsum_sym (one block per column): initialises work[0..n-1] with
         *   the T1 and T2 triangular column sums (including symmetric reflection).
         * k_colsum_rect (one block per max(dim1,dim2)): adds S contributions
         *   via atomicAdd.
         *
         * Final max via cublasIsamax on work[0..n-1].
         */
        CHK_CU(cudaMallocAsync((void **)&d_work, (size_t)n * sizeof(float), stream));

        size_t smem = BLOCK * sizeof(float);
        int upper1 = (p.fill1 == CUBLAS_FILL_MODE_UPPER) ? 1 : 0;
        int upper2 = (p.fill2 == CUBLAS_FILL_MODE_UPPER) ? 1 : 0;

        /* T1: blocks 0..dim1-1, writing work[0..dim1-1] */
        if (p.dim1 > 0) {
            k_colsum_sym<<<p.dim1, BLOCK, smem, stream>>>(
                arf, p.off1, p.dim1, p.lda1, upper1, 0, d_work);
            CHK_CU(cudaGetLastError());
        }
        /* T2: blocks 0..dim2-1, writing work[dim1..n-1] */
        if (p.dim2 > 0) {
            k_colsum_sym<<<p.dim2, BLOCK, smem, stream>>>(
                arf, p.off2, p.dim2, p.lda2, upper2, p.dim1, d_work);
            CHK_CU(cudaGetLastError());
        }

        /* S block: add contributions via atomicAdd */
        if (p.dim1 > 0 && p.dim2 > 0) {
            int is_a21 = (p.s_op1 == CUBLAS_OP_T) ? 1 : 0;
            int rows   = is_a21 ? p.dim2 : p.dim1;
            int cols   = is_a21 ? p.dim1 : p.dim2;
            int grid   = max(rows, cols);
            k_colsum_rect<<<grid, BLOCK, smem, stream>>>(
                arf + p.s_off, rows, cols, p.s_lda, is_a21, p.dim1, d_work);
            CHK_CU(cudaGetLastError());
        }

        int idx = 0;
        CHK_CB(cublasIsamax(cb, n, d_work, 1, &idx));
        float val = 0.0f;
        CHK_CU(cudaMemcpyAsync(&val, d_work + (idx - 1), sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
        CHK_CU(cudaStreamSynchronize(stream));
        *result = val;   /* already non-negative (fabsf-based column sums) */

    } else if (norm == CURFP_NORM_FRO) {
        /*
         * ||A||_F = sqrt(2·||arf||² − ||diag(T1)||² − ||diag(T2)||²)
         *
         * Uses cublasSnrm2 (not cublasSdot) to avoid float32 catastrophic
         * accumulation at large n.  cublasSdot on n*(n+1)/2 ~ 500M float32
         * elements loses all precision; cublasSnrm2 uses a stable algorithm.
         */
        if (nt < 0) { status = CURFP_STATUS_INVALID_VALUE; goto cleanup; }

        float arf_nrm  = 0.0f;
        float diag1_nrm = 0.0f;
        float diag2_nrm = 0.0f;

        /* ||arf|| via stable cublasSnrm2 */
        CHK_CB(cublasSnrm2(cb, nt, arf, 1, &arf_nrm));

        /* ||diag(T1)|| — stride lda1+1 picks the diagonal elements */
        if (p.dim1 > 0)
            CHK_CB(cublasSnrm2(cb, p.dim1, arf + p.off1, p.lda1 + 1, &diag1_nrm));

        /* ||diag(T2)|| — stride lda2+1 picks the diagonal elements */
        if (p.dim2 > 0)
            CHK_CB(cublasSnrm2(cb, p.dim2, arf + p.off2, p.lda2 + 1, &diag2_nrm));

        float fro_sq = 2.0f * arf_nrm  * arf_nrm
                     -        diag1_nrm * diag1_nrm
                     -        diag2_nrm * diag2_nrm;
        *result = (fro_sq > 0.0f) ? sqrtf(fro_sq) : 0.0f;

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
