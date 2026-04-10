/*
 * curfpSstrttf — Copy triangular matrix from standard full to RFP format
 *
 * Direct CUDA translation of LAPACK strttf.f.
 * A is an n×n row-major matrix on device (lda >= n).
 * ARF is n*(n+1)/2 floats on device.
 *
 * 8 RFP storage variants (N parity × TRANSR N/T × UPLO L/U).
 * Each kernel parallelises over the outer loop variable J (one thread per J).
 *
 * LAPACK column-major A(row,col) maps to row-major A[row*lda+col] here.
 * The triangular element set is identical; only the in-memory stride differs.
 *
 * Cases 4, 7, 8 previously required two sequential kernel launches; they have
 * been fused into single kernels (no data dependencies between the sub-regions).
 */

#include "curfp_internal.h"

/* Row-major element index */
#define A_RM(r, c, lda) ((r) * (lda) + (c))

/* Check last kernel-launch error and return on failure */
#define KL_CHECK() \
    do { \
        cudaError_t _ke = cudaGetLastError(); \
        if (_ke != cudaSuccess) return CURFP_STATUS_EXECUTION_FAILED; \
    } while (0)

static const int BLOCK = 256;

/* ================================================================
 * Case 1: N odd, TRANSR='N', UPLO='L'
 * ================================================================ */
static __global__ void k_strttf_odd_N_L(const float* __restrict__ A,
                                         float* __restrict__ arf,
                                         int n, int n1, int n2, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > n2) return;

    for (int i = n1; i <= n2 + j; i++)
        arf[j * n + (i - n1)] = A[A_RM(n2 + j, i, lda)];

    for (int i = j; i < n; i++)
        arf[j * n + i] = A[A_RM(i, j, lda)];
}

/* ================================================================
 * Case 2: N odd, TRANSR='N', UPLO='U'
 * ================================================================ */
static __global__ void k_strttf_odd_N_U(const float* __restrict__ A,
                                         float* __restrict__ arf,
                                         int n, int n1, int n2,
                                         int nt, int lda)
{
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (j_idx >= n2) return;

    int j = n - 1 - j_idx;
    int ij0 = nt - (n - j) * n;

    for (int i = 0; i <= j; i++)
        arf[ij0 + i] = A[A_RM(i, j, lda)];

    for (int l = j - n1; l <= n1 - 1; l++)
        arf[ij0 + (j + 1) + (l - (j - n1))] = A[A_RM(j - n1, l, lda)];
}

/* ================================================================
 * Case 3: N odd, TRANSR='T', UPLO='L'
 * ================================================================ */
static __global__ void k_strttf_odd_T_L(const float* __restrict__ A,
                                         float* __restrict__ arf,
                                         int n, int n1, int n2, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int ij0 = j * n1;

    if (j < n2) {
        for (int i = 0; i <= j; i++)
            arf[ij0 + i] = A[A_RM(j, i, lda)];
        for (int i = n1 + j; i < n; i++)
            arf[ij0 + (i - n1 + 1)] = A[A_RM(i, n1 + j, lda)];
    } else {
        for (int i = 0; i < n1; i++)
            arf[ij0 + i] = A[A_RM(j, i, lda)];
    }
}

/* ================================================================
 * Case 4: N odd, TRANSR='T', UPLO='U'  [fused]
 * N1 = floor(N/2), N2 = ceil(N/2) = N1+1.
 * Grid: N1+1 threads (covers both former A and B sub-loops).
 *
 * Former kernel A — j=0..N1, IJ_start = j*N2:
 *   I=N1..N-1: arf[j*N2 + I-N1] = A[j, I]
 * Former kernel B — j=0..N1-1, IJ_start = (N1+1)*N2 + j*(N1+1):
 *   Part1 (I=0..j):       arf[IJ + I]                = A[I, j]
 *   Part2 (L=N2+j..N-1):  arf[IJ + (j+1) + L-(N2+j)] = A[N2+j, L]
 * ================================================================ */
static __global__ void k_strttf_odd_T_U(const float* __restrict__ A,
                                          float* __restrict__ arf,
                                          int n, int n1, int n2, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > n1) return;

    /* A-part */
    int ij0 = j * n2;
    for (int i = n1; i < n; i++)
        arf[ij0 + (i - n1)] = A[A_RM(j, i, lda)];

    /* B-part */
    if (j < n1) {
        int ij1 = (n1 + 1) * n2 + j * (n1 + 1);
        for (int i = 0; i <= j; i++)
            arf[ij1 + i] = A[A_RM(i, j, lda)];
        for (int l = n2 + j; l < n; l++)
            arf[ij1 + (j + 1) + (l - (n2 + j))] = A[A_RM(n2 + j, l, lda)];
    }
}

/* ================================================================
 * Case 5: N even, TRANSR='N', UPLO='L'
 * ================================================================ */
static __global__ void k_strttf_even_N_L(const float* __restrict__ A,
                                          float* __restrict__ arf,
                                          int n, int k, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;

    int ij0 = j * (n + 1);

    for (int i = k; i <= k + j; i++)
        arf[ij0 + (i - k)] = A[A_RM(k + j, i, lda)];

    for (int i = j; i < n; i++)
        arf[ij0 + i + 1] = A[A_RM(i, j, lda)];
}

/* ================================================================
 * Case 6: N even, TRANSR='N', UPLO='U'
 * ================================================================ */
static __global__ void k_strttf_even_N_U(const float* __restrict__ A,
                                          float* __restrict__ arf,
                                          int n, int k, int nt, int lda)
{
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + j_idx;
    if (j >= n) return;

    int ij0 = nt - (n + 1) * (n - j);

    for (int i = 0; i <= j; i++)
        arf[ij0 + i] = A[A_RM(i, j, lda)];

    for (int l = j - k; l <= k - 1; l++)
        arf[ij0 + k + 1 + l] = A[A_RM(j - k, l, lda)];
}

/* ================================================================
 * Case 7: N even, TRANSR='T', UPLO='L'  [fused]
 * K = N/2.
 * Grid: N threads.
 *
 * Former kernel col (j=0..K-1): arf[j] = A[K+j, K]
 * Former kernel main (j=0..N-1), IJ_start = K*(j+1):
 *   j < K-1:  arf[IJ+i]=A[j,i] (i=0..j);  arf[IJ+i-K]=A[i,K+1+j] (i=K+1+j..N-1)
 *   j >= K-1: arf[IJ+i]=A[j,i] (i=0..K-1)
 *
 * Col and main write disjoint arf regions (col: arf[0..K-1], main: arf[K..]).
 * ================================================================ */
static __global__ void k_strttf_even_T_L(const float* __restrict__ A,
                                           float* __restrict__ arf,
                                           int n, int k, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    /* col part: arf[0..K-1] = A[K..N-1, K] */
    if (j < k)
        arf[j] = A[A_RM(k + j, k, lda)];

    /* main part */
    int ij0 = k * (j + 1);
    if (j < k - 1) {
        for (int i = 0; i <= j; i++)
            arf[ij0 + i] = A[A_RM(j, i, lda)];
        for (int i = k + 1 + j; i < n; i++)
            arf[ij0 + (i - k)] = A[A_RM(i, k + 1 + j, lda)];
    } else {
        for (int i = 0; i < k; i++)
            arf[ij0 + i] = A[A_RM(j, i, lda)];
    }
}

/* ================================================================
 * Case 8: N even, TRANSR='T', UPLO='U'  [fused]
 * K = N/2.
 * Grid: K+1 threads.
 *
 * Former kernel A — j=0..K, IJ_start = j*K:
 *   I=K..N-1: arf[j*K + I-K] = A[j, I]
 * Former kernel B — j=0..K-1, IJ_start = K*(K+1) + j*K:
 *   Part1 (I=0..j):        arf[IJ + I]                   = A[I, j]
 *   Part2 (L=K+1+j..N-1):  arf[IJ + (j+1) + L-(K+1+j)] = A[K+1+j, L]
 * ================================================================ */
static __global__ void k_strttf_even_T_U(const float* __restrict__ A,
                                           float* __restrict__ arf,
                                           int n, int k, int lda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > k) return;

    /* A-part */
    int ij0 = j * k;
    for (int i = k; i < n; i++)
        arf[ij0 + (i - k)] = A[A_RM(j, i, lda)];

    /* B-part */
    if (j < k) {
        int ij1 = k * (k + 1) + j * k;
        for (int i = 0; i <= j; i++)
            arf[ij1 + i] = A[A_RM(i, j, lda)];
        for (int l = k + 1 + j; l < n; l++)
            arf[ij1 + (j + 1) + (l - (k + 1 + j))] = A[A_RM(k + 1 + j, l, lda)];
    }
}

/* ================================================================
 * Public entry point
 * ================================================================ */
extern "C"
curfpStatus_t curfpSstrttf(curfpHandle_t    handle,
                            curfpOperation_t transr,
                            curfpFillMode_t  uplo,
                            int              n,
                            const float     *A,
                            int              lda,
                            float           *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || lda < n) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    /* n=1 trivial */
    if (n == 1) {
        CURFP_CHECK_CUDA(cudaMemcpy(arf, A, sizeof(float),
                                    cudaMemcpyDeviceToDevice));
        return CURFP_STATUS_SUCCESS;
    }

    cudaStream_t stream;
    CURFP_CHECK_CUBLAS(cublasGetStream(handle->cublas, &stream));

    int nisodd       = n % 2;
    int normaltransr = (transr == CURFP_OP_N);
    int lower        = (uplo == CURFP_FILL_MODE_LOWER);

    /* Sub-block dimensions (same as LAPACK strttf.f) */
    int n1, n2, k = 0, nt;
    if (lower) { n2 = n / 2; n1 = n - n2; }
    else       { n1 = n / 2; n2 = n - n1; }
    if (!nisodd) k = n / 2;
    nt = n * (n + 1) / 2;

    /* Grid helpers */
    auto grid = [](int threads) {
        return (threads + BLOCK - 1) / BLOCK;
    };

    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                /* Case 1 */
                k_strttf_odd_N_L<<<grid(n2 + 1), BLOCK, 0, stream>>>(
                    A, arf, n, n1, n2, lda);
            } else {
                /* Case 2 */
                k_strttf_odd_N_U<<<grid(n2), BLOCK, 0, stream>>>(
                    A, arf, n, n1, n2, nt, lda);
            }
            KL_CHECK();
        } else {
            if (lower) {
                /* Case 3 */
                k_strttf_odd_T_L<<<grid(n), BLOCK, 0, stream>>>(
                    A, arf, n, n1, n2, lda);
                KL_CHECK();
            } else {
                /* Case 4 (fused) */
                k_strttf_odd_T_U<<<grid(n1 + 1), BLOCK, 0, stream>>>(
                    A, arf, n, n1, n2, lda);
                KL_CHECK();
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                /* Case 5 */
                k_strttf_even_N_L<<<grid(k), BLOCK, 0, stream>>>(
                    A, arf, n, k, lda);
            } else {
                /* Case 6 */
                k_strttf_even_N_U<<<grid(k), BLOCK, 0, stream>>>(
                    A, arf, n, k, nt, lda);
            }
            KL_CHECK();
        } else {
            if (lower) {
                /* Case 7 (fused) */
                k_strttf_even_T_L<<<grid(n), BLOCK, 0, stream>>>(
                    A, arf, n, k, lda);
                KL_CHECK();
            } else {
                /* Case 8 (fused) */
                k_strttf_even_T_U<<<grid(k + 1), BLOCK, 0, stream>>>(
                    A, arf, n, k, lda);
                KL_CHECK();
            }
        }
    }

    return CURFP_STATUS_SUCCESS;
}
