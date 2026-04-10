/*
 * curfpSstfttr — Copy triangular matrix from RFP format to standard full format
 *
 * Inverse of curfpSstrttf: direct CUDA translation of LAPACK stfttr.f.
 * ARF is n*(n+1)/2 floats on device.
 * A is an n×n row-major matrix on device (lda >= n).
 * Only the triangle specified by UPLO is written; the other triangle
 * is not touched (caller should zero-initialize A if needed).
 *
 * 8 RFP storage variants (N parity × TRANSR N/T × UPLO L/U).
 * Kernels are identical to curfp_strttf.cu with src/dst swapped.
 *
 * Cases 4, 7, 8 previously required two sequential kernel launches; they have
 * been fused into single kernels (no data dependencies between the sub-regions).
 */

#include "curfp_internal.h"

#define A_RM(r, c, lda) ((r) * (lda) + (c))

#define KL_CHECK() \
    do { \
        cudaError_t _ke = cudaGetLastError(); \
        if (_ke != cudaSuccess) return CURFP_STATUS_EXECUTION_FAILED; \
    } while (0)

static const int BLOCK = 256;

/* ================================================================ Case 1 */
static __global__ void k_stfttr_odd_N_L(const float* __restrict__ arf,
                                         float* __restrict__ A,
                                         int n, int n1, int n2, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > n2) return;

    for (int i = n1; i <= n2 + j; i++)
        A[A_RM(n2 + j, i, lda)] = arf[j * n + (i - n1)];

    for (int i = j; i < n; i++)
        A[A_RM(i, j, lda)] = arf[j * n + i];
}

/* ================================================================ Case 2 */
static __global__ void k_stfttr_odd_N_U(const float* __restrict__ arf,
                                         float* __restrict__ A,
                                         int n, int n1, int n2,
                                         int nt, int lda)
{
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (j_idx >= n2) return;

    int j = n - 1 - j_idx;
    int ij0 = nt - (n - j) * n;

    for (int i = 0; i <= j; i++)
        A[A_RM(i, j, lda)] = arf[ij0 + i];

    for (int l = j - n1; l <= n1 - 1; l++)
        A[A_RM(j - n1, l, lda)] = arf[ij0 + (j + 1) + (l - (j - n1))];
}

/* ================================================================ Case 3 */
static __global__ void k_stfttr_odd_T_L(const float* __restrict__ arf,
                                         float* __restrict__ A,
                                         int n, int n1, int n2, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int ij0 = j * n1;

    if (j < n2) {
        for (int i = 0; i <= j; i++)
            A[A_RM(j, i, lda)] = arf[ij0 + i];
        for (int i = n1 + j; i < n; i++)
            A[A_RM(i, n1 + j, lda)] = arf[ij0 + (i - n1 + 1)];
    } else {
        for (int i = 0; i < n1; i++)
            A[A_RM(j, i, lda)] = arf[ij0 + i];
    }
}

/* ================================================================ Case 4 [fused] */
static __global__ void k_stfttr_odd_T_U(const float* __restrict__ arf,
                                          float* __restrict__ A,
                                          int n, int n1, int n2, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > n1) return;

    /* A-part */
    int ij0 = j * n2;
    for (int i = n1; i < n; i++)
        A[A_RM(j, i, lda)] = arf[ij0 + (i - n1)];

    /* B-part */
    if (j < n1) {
        int ij1 = (n1 + 1) * n2 + j * (n1 + 1);
        for (int i = 0; i <= j; i++)
            A[A_RM(i, j, lda)] = arf[ij1 + i];
        for (int l = n2 + j; l < n; l++)
            A[A_RM(n2 + j, l, lda)] = arf[ij1 + (j + 1) + (l - (n2 + j))];
    }
}

/* ================================================================ Case 5 */
static __global__ void k_stfttr_even_N_L(const float* __restrict__ arf,
                                          float* __restrict__ A,
                                          int n, int k, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;

    int ij0 = j * (n + 1);

    for (int i = k; i <= k + j; i++)
        A[A_RM(k + j, i, lda)] = arf[ij0 + (i - k)];

    for (int i = j; i < n; i++)
        A[A_RM(i, j, lda)] = arf[ij0 + i + 1];
}

/* ================================================================ Case 6 */
static __global__ void k_stfttr_even_N_U(const float* __restrict__ arf,
                                          float* __restrict__ A,
                                          int n, int k, int nt, int lda)
{
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + j_idx;
    if (j >= n) return;

    int ij0 = nt - (n + 1) * (n - j);

    for (int i = 0; i <= j; i++)
        A[A_RM(i, j, lda)] = arf[ij0 + i];

    for (int l = j - k; l <= k - 1; l++)
        A[A_RM(j - k, l, lda)] = arf[ij0 + k + 1 + l];
}

/* ================================================================ Case 7 [fused] */
static __global__ void k_stfttr_even_T_L(const float* __restrict__ arf,
                                           float* __restrict__ A,
                                           int n, int k, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    /* col part: A[K+j, K] = arf[j]  (j=0..K-1) */
    if (j < k)
        A[A_RM(k + j, k, lda)] = arf[j];

    /* main part */
    int ij0 = k * (j + 1);
    if (j < k - 1) {
        for (int i = 0; i <= j; i++)
            A[A_RM(j, i, lda)] = arf[ij0 + i];
        for (int i = k + 1 + j; i < n; i++)
            A[A_RM(i, k + 1 + j, lda)] = arf[ij0 + (i - k)];
    } else {
        for (int i = 0; i < k; i++)
            A[A_RM(j, i, lda)] = arf[ij0 + i];
    }
}

/* ================================================================ Case 8 [fused] */
static __global__ void k_stfttr_even_T_U(const float* __restrict__ arf,
                                           float* __restrict__ A,
                                           int n, int k, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > k) return;

    /* A-part */
    int ij0 = j * k;
    for (int i = k; i < n; i++)
        A[A_RM(j, i, lda)] = arf[ij0 + (i - k)];

    /* B-part */
    if (j < k) {
        int ij1 = k * (k + 1) + j * k;
        for (int i = 0; i <= j; i++)
            A[A_RM(i, j, lda)] = arf[ij1 + i];
        for (int l = k + 1 + j; l < n; l++)
            A[A_RM(k + 1 + j, l, lda)] = arf[ij1 + (j + 1) + (l - (k + 1 + j))];
    }
}

/* ================================================================
 * Public entry point
 * ================================================================ */
extern "C"
curfpStatus_t curfpSstfttr(curfpHandle_t    handle,
                            curfpOperation_t transr,
                            curfpFillMode_t  uplo,
                            int              n,
                            const float     *arf,
                            float           *A,
                            int              lda)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || lda < n) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    if (n == 1) {
        CURFP_CHECK_CUDA(cudaMemcpy(A, arf, sizeof(float),
                                    cudaMemcpyDeviceToDevice));
        return CURFP_STATUS_SUCCESS;
    }

    cudaStream_t stream;
    CURFP_CHECK_CUBLAS(cublasGetStream(handle->cublas, &stream));

    int nisodd       = n % 2;
    int normaltransr = (transr == CURFP_OP_N);
    int lower        = (uplo == CURFP_FILL_MODE_LOWER);

    int n1, n2, k = 0, nt;
    if (lower) { n2 = n / 2; n1 = n - n2; }
    else       { n1 = n / 2; n2 = n - n1; }
    if (!nisodd) k = n / 2;
    nt = n * (n + 1) / 2;

    auto grid = [](int threads) {
        return (threads + BLOCK - 1) / BLOCK;
    };

    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                k_stfttr_odd_N_L<<<grid(n2 + 1), BLOCK, 0, stream>>>(
                    arf, A, n, n1, n2, lda);
            } else {
                k_stfttr_odd_N_U<<<grid(n2), BLOCK, 0, stream>>>(
                    arf, A, n, n1, n2, nt, lda);
            }
            KL_CHECK();
        } else {
            if (lower) {
                k_stfttr_odd_T_L<<<grid(n), BLOCK, 0, stream>>>(
                    arf, A, n, n1, n2, lda);
                KL_CHECK();
            } else {
                /* Case 4 (fused) */
                k_stfttr_odd_T_U<<<grid(n1 + 1), BLOCK, 0, stream>>>(
                    arf, A, n, n1, n2, lda);
                KL_CHECK();
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                k_stfttr_even_N_L<<<grid(k), BLOCK, 0, stream>>>(
                    arf, A, n, k, lda);
            } else {
                k_stfttr_even_N_U<<<grid(k), BLOCK, 0, stream>>>(
                    arf, A, n, k, nt, lda);
            }
            KL_CHECK();
        } else {
            if (lower) {
                /* Case 7 (fused) */
                k_stfttr_even_T_L<<<grid(n), BLOCK, 0, stream>>>(
                    arf, A, n, k, lda);
                KL_CHECK();
            } else {
                /* Case 8 (fused) */
                k_stfttr_even_T_U<<<grid(k + 1), BLOCK, 0, stream>>>(
                    arf, A, n, k, lda);
                KL_CHECK();
            }
        }
    }

    return CURFP_STATUS_SUCCESS;
}
