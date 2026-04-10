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
 *   Accumulate absolute column sums from each block into a work vector of
 *   length n using atomicAdd, then take the max.
 *
 * Frobenius norm:
 *   ||A||_F² = 2·||arf||² − ||diag(T1)||² − ||diag(T2)||²
 *   because each off-diagonal element appears once in arf but contributes
 *   twice to ||A||_F² (A is symmetric, so A[i,j]=A[j,i]).
 *
 * Max-element norm:
 *   cublasIsamax on the full arf array (every stored element is a unique A[i,j]).
 */

#include <math.h>
#include <climits>
#include "curfp_internal.h"

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

/* -------------------------------------------------------------------------
 * Kernels for 1-norm computation — column-parallel, no atomic operations.
 *
 * Strategy: one thread per column of each sub-block.  Each thread independently
 * accumulates the full column absolute sum (including the reflected symmetric
 * half) and writes to a unique work[] slot → zero atomic contention.
 * ------------------------------------------------------------------------- */

/*
 * k_colsums_sym: compute column absolute sums for both triangular sub-blocks T1 and T2.
 *
 * T1 (dim1×dim1, fill=upper1, lda=lda1, base offset off1 into arf):
 *   Column j (j<dim1): stored half (stride-1 down the column) + reflected half
 *   (stride-lda1 across rows, using A[r,j]=A[j,r] symmetry) → work[j]
 *
 * T2 (dim2×dim2, fill=upper2, lda=lda2, base offset off2 into arf):
 *   Same formula → work[dim1+j]
 *
 * No atomics: thread j writes exclusively to work[j] and work[dim1+j].
 * Initialises work[] directly; no prior memset required.
 */
static __global__ void k_colsums_sym(
    const float * __restrict__ arf,
    long off1, int dim1, int lda1, int upper1,
    long off2, int dim2, int lda2, int upper2,
    float * __restrict__ work)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const float *T1 = arf + off1;
    const float *T2 = arf + off2;

    if (j < dim1) {
        float s = 0.0f;
        if (upper1) {
            /* stored: rows 0..j at T1[j*lda1 + r]; reflected: rows j+1..dim1-1 */
            for (int r = 0;   r <= j;   r++) s += fabsf(T1[(long)j*lda1 + r]);
            for (int r = j+1; r < dim1; r++) s += fabsf(T1[(long)r*lda1 + j]);
        } else {
            /* stored: rows j..dim1-1 at T1[j*lda1 + r]; reflected: rows 0..j-1 */
            for (int r = j;   r < dim1; r++) s += fabsf(T1[(long)j*lda1 + r]);
            for (int r = 0;   r < j;    r++) s += fabsf(T1[(long)r*lda1 + j]);
        }
        work[j] = s;
    }

    if (j < dim2) {
        float s = 0.0f;
        if (upper2) {
            for (int r = 0;   r <= j;   r++) s += fabsf(T2[(long)j*lda2 + r]);
            for (int r = j+1; r < dim2; r++) s += fabsf(T2[(long)r*lda2 + j]);
        } else {
            for (int r = j;   r < dim2; r++) s += fabsf(T2[(long)j*lda2 + r]);
            for (int r = 0;   r < j;    r++) s += fabsf(T2[(long)r*lda2 + j]);
        }
        work[dim1 + j] = s;
    }
}

/*
 * k_colsums_rect: accumulate S block contributions into work[] (no atomics).
 *
 * S is column-major, shape rows×cols, pointer = arf + s_off, lda = s_lda.
 *
 * is_a21=1: S = A[dim1:n, 0:dim1]  (dim2 rows × dim1 cols)
 *   col j of S (j<dim1) → sum over rows → work[j]      += col_sum
 *   row j of S (j<dim2) → sum over cols → work[dim1+j] += row_sum
 *
 * is_a21=0: S = A[0:dim1, dim1:n]  (dim1 rows × dim2 cols)
 *   row j of S (j<dim1) → sum over cols → work[j]      += row_sum
 *   col j of S (j<dim2) → sum over rows → work[dim1+j] += col_sum
 *
 * No atomics: for any j, the two writes go to work[j] and work[dim1+j] —
 * these slots are disjoint across threads and disjoint from each other.
 * Must run after k_colsums_sym (which initialises work[]).
 */
static __global__ void k_colsums_rect(
    const float * __restrict__ S,
    int rows, int cols, int lda,
    int is_a21, int dim1,
    float * __restrict__ work)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (is_a21) {
        /* col j (j<cols=dim1) → work[j];  row j (j<rows=dim2) → work[dim1+j] */
        if (j < cols) {
            float s = 0.0f;
            for (int r = 0; r < rows; r++) s += fabsf(S[(long)j*lda + r]);
            work[j] += s;
        }
        if (j < rows) {
            float s = 0.0f;
            for (int c = 0; c < cols; c++) s += fabsf(S[(long)c*lda + j]);
            work[dim1 + j] += s;
        }
    } else {
        /* row j (j<rows=dim1) → work[j];  col j (j<cols=dim2) → work[dim1+j] */
        if (j < rows) {
            float s = 0.0f;
            for (int c = 0; c < cols; c++) s += fabsf(S[(long)c*lda + j]);
            work[j] += s;
        }
        if (j < cols) {
            float s = 0.0f;
            for (int r = 0; r < rows; r++) s += fabsf(S[(long)j*lda + r]);
            work[dim1 + j] += s;
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
     * cublasSdot / cublasIsamax take int n, so cap at INT_MAX; -1 means "too large". */
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
        CHK_CU(cudaMemcpy(&val, arf + (idx - 1), sizeof(float),
                          cudaMemcpyDeviceToHost));
        *result = fabsf(val);

    } else if (norm == CURFP_NORM_ONE) {
        /* max column absolute sum — column-parallel kernels, no atomic operations.
         * k_colsums_sym  writes  work[0..n-1] (initialises; no memset needed).
         * k_colsums_rect adds S contributions; must run after k_colsums_sym. */
        CHK_CU(cudaMallocAsync((void **)&d_work, (size_t)n * sizeof(float), stream));

        int grid1 = (max(p.dim1, p.dim2) + BLOCK - 1) / BLOCK;
        int upper1 = (p.fill1 == CUBLAS_FILL_MODE_UPPER) ? 1 : 0;
        int upper2 = (p.fill2 == CUBLAS_FILL_MODE_UPPER) ? 1 : 0;
        k_colsums_sym<<<grid1, BLOCK, 0, stream>>>(
            arf, p.off1, p.dim1, p.lda1, upper1,
                 p.off2, p.dim2, p.lda2, upper2,
            d_work);
        CHK_CU(cudaGetLastError());

        if (p.dim1 > 0 && p.dim2 > 0) {
            int is_a21 = (p.s_op1 == CUBLAS_OP_T) ? 1 : 0;
            int rows   = is_a21 ? p.dim2 : p.dim1;
            int cols   = is_a21 ? p.dim1 : p.dim2;
            int grid2  = (max(rows, cols) + BLOCK - 1) / BLOCK;
            k_colsums_rect<<<grid2, BLOCK, 0, stream>>>(
                arf + p.s_off, rows, cols, p.s_lda, is_a21, p.dim1, d_work);
            CHK_CU(cudaGetLastError());
        }

        int idx = 0;
        CHK_CB(cublasIsamax(cb, n, d_work, 1, &idx));
        float val = 0.0f;
        CHK_CU(cudaMemcpy(&val, d_work + (idx - 1), sizeof(float),
                          cudaMemcpyDeviceToHost));
        *result = val;   /* already non-negative (fabsf-based column sums) */

    } else if (norm == CURFP_NORM_FRO) {
        /*
         * ||A||_F = sqrt(2·||arf||² − ||diag(T1)||² − ||diag(T2)||²)
         *
         * Derivation: each off-diagonal stored element contributes ×2 to ||A||_F²
         * (once from A[i,j], once from A[j,i]).  So:
         *   ||A||_F² = 2·||arf||² − diag_sq(T1) − diag_sq(T2)
         */
        if (nt < 0) { status = CURFP_STATUS_INVALID_VALUE; goto cleanup; }
        float arf_sq   = 0.0f;
        float diag1_sq = 0.0f;
        float diag2_sq = 0.0f;
        CHK_CB(cublasSdot(cb, nt, arf, 1, arf, 1, &arf_sq));
        if (p.dim1 > 0)
            CHK_CB(cublasSdot(cb, p.dim1, arf + p.off1, p.lda1 + 1,
                              arf + p.off1, p.lda1 + 1, &diag1_sq));
        if (p.dim2 > 0)
            CHK_CB(cublasSdot(cb, p.dim2, arf + p.off2, p.lda2 + 1,
                              arf + p.off2, p.lda2 + 1, &diag2_sq));
        float fro_sq = 2.0f * arf_sq - diag1_sq - diag2_sq;
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
