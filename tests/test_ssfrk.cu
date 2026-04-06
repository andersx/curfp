/*
 * test_ssfrk.cu — Correctness tests for curfpSsfrk
 *
 * Tests:
 *   1. beta=0 quick-return: all RFP variants, result should be zero when k=0.
 *   2. beta=1, alpha=0 pass-through: C unchanged.
 *   3. Numerical: n=2, TRANSR=N, UPLO=L, TRANS=N — hand-verified rank-1 update.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "curfp.h"

/* =========================================================================
 * Error-checking macros
 * =========================================================================*/
#define CHECK_CUDA(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

#define CHECK_CURFP(x) do { \
    curfpStatus_t _s = (x); \
    if (_s != CURFP_STATUS_SUCCESS) { \
        fprintf(stderr, "curfp error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
        return 1; \
    } \
} while (0)

/* =========================================================================
 * Test 1: beta=0, k=0 → all RFP elements should become 0
 * =========================================================================*/
static int test_beta_zero(curfpHandle_t handle, int n,
                           curfpOperation_t transr, curfpFillMode_t uplo)
{
    int nt = n * (n + 1) / 2;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)nt * sizeof(float)));

    /* Fill with sentinel 99 */
    float *h_C = (float *)malloc((size_t)nt * sizeof(float));
    for (int i = 0; i < nt; i++) h_C[i] = 99.0f;
    CHECK_CUDA(cudaMemcpy(d_C, h_C, (size_t)nt * sizeof(float), cudaMemcpyHostToDevice));
    free(h_C);

    /* alpha=0, beta=0 → hits the cudaMemset path, C becomes zero.
     * A=NULL is safe because the implementation returns before accessing A. */
    float alpha = 0.0f, beta = 0.0f;
    CHECK_CURFP(curfpSsfrk(handle, transr, uplo, CURFP_OP_N,
                            n, 1, &alpha, NULL, n, &beta, d_C));

    float *h_res = (float *)malloc((size_t)nt * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_res, d_C, (size_t)nt * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_C);

    int ok = 1;
    for (int i = 0; i < nt; i++) {
        if (fabsf(h_res[i]) > 1e-5f) { ok = 0; break; }
    }
    free(h_res);
    return ok ? 0 : 1;
}

/* =========================================================================
 * Test 2: alpha=0, beta=1 → C should be unchanged
 * =========================================================================*/
static int test_alpha_zero_beta_one(curfpHandle_t handle, int n,
                                    curfpOperation_t transr, curfpFillMode_t uplo)
{
    int nt = n * (n + 1) / 2;
    float *h_C = (float *)malloc((size_t)nt * sizeof(float));
    srand(42 + n);
    for (int i = 0; i < nt; i++) h_C[i] = (float)(rand() % 100) / 10.0f;

    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)nt * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, (size_t)nt * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 0.0f, beta = 1.0f;
    /* A can be anything since alpha=0; pass NULL with lda=n (ignored) */
    CHECK_CURFP(curfpSsfrk(handle, transr, uplo, CURFP_OP_N,
                            n, 3, &alpha, NULL, n, &beta, d_C));

    float *h_res = (float *)malloc((size_t)nt * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_res, d_C, (size_t)nt * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_C);

    int ok = 1;
    for (int i = 0; i < nt; i++) {
        if (fabsf(h_res[i] - h_C[i]) > 1e-5f) { ok = 0; break; }
    }
    free(h_C);
    free(h_res);
    return ok ? 0 : 1;
}

/* =========================================================================
 * Test 3: n=2, TRANSR=N, UPLO=L, TRANS=N
 *
 * RFP layout for n=2 even, TRANSR=N, UPLO=L (Case 5 from plan):
 *   L11 block at ARF+1 (1x1, lower, lda_rfp=3): ARF[1] = C(0,0)
 *   L22 block at ARF+0 (1x1, upper, lda_rfp=3): ARF[0] = C(1,1)
 *   Off-diag at ARF+2 (1x1):                    ARF[2] = C(1,0)
 *
 * So: ARF = { C(1,1), C(0,0), C(1,0) }
 *
 * We set:
 *   alpha=2, beta=0.5
 *   A = [a0, a1]^T  (2x1, col-major, lda=2)
 *   C_init = [[c00, c10], [c10, c11]]
 *
 * Expected result:
 *   C(i,j) = alpha * A[i]*A[j] + beta * C_init(i,j)
 * =========================================================================*/
static int test_n2_NL_rankk(curfpHandle_t handle)
{
    const float a0 = 2.0f, a1 = 3.0f;
    const float c00 = 1.0f, c10 = 0.5f, c11 = 4.0f;
    const float alpha = 2.0f, beta = 0.5f;

    const float ref00 = alpha * a0 * a0 + beta * c00;
    const float ref10 = alpha * a1 * a0 + beta * c10;
    const float ref11 = alpha * a1 * a1 + beta * c11;

    /* RFP encoding: ARF[0]=c11, ARF[1]=c00, ARF[2]=c10 */
    float h_C[3] = { c11, c00, c10 };
    /* A is 2x1 col-major, lda=2: h_A[0]=a0, h_A[1]=a1 */
    float h_A[2] = { a0, a1 };

    float *d_C, *d_A;
    CHECK_CUDA(cudaMalloc(&d_C, 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A, 2 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, 2 * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CURFP(curfpSsfrk(handle, CURFP_OP_N, CURFP_FILL_MODE_LOWER,
                            CURFP_OP_N, 2, 1, &alpha, d_A, 2, &beta, d_C));

    float h_result[3];
    CHECK_CUDA(cudaMemcpy(h_result, d_C, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_C);
    cudaFree(d_A);

    /* Decode: h_result[1]=C(0,0), h_result[2]=C(1,0), h_result[0]=C(1,1) */
    const float tol = 1e-4f;
    int ok = (fabsf(h_result[1] - ref00) < tol) &&
             (fabsf(h_result[2] - ref10) < tol) &&
             (fabsf(h_result[0] - ref11) < tol);

    if (!ok) {
        printf("\n  Expected: C(0,0)=%.4f C(1,0)=%.4f C(1,1)=%.4f",
               ref00, ref10, ref11);
        printf("\n  Got:      C(0,0)=%.4f C(1,0)=%.4f C(1,1)=%.4f",
               h_result[1], h_result[2], h_result[0]);
    }
    return ok ? 0 : 1;
}

/* =========================================================================
 * Test 4: n=1 (trivial: single diagonal element)
 * alpha=3, beta=2, k=1, A=[a], C=[c]
 * Result: 3*a*a + 2*c
 *
 * For n=1: all 4 transr/uplo variants map to the same single-element array.
 * =========================================================================*/
static int test_n1_rankk(curfpHandle_t handle,
                          curfpOperation_t transr, curfpFillMode_t uplo)
{
    const float a = 3.0f, c = 5.0f;
    const float alpha = 2.0f, beta = 0.5f;
    const float expected = alpha * a * a + beta * c;

    float *d_C, *d_A;
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_C, &c, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A, &a, sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CURFP(curfpSsfrk(handle, transr, uplo, CURFP_OP_N,
                            1, 1, &alpha, d_A, 1, &beta, d_C));

    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_C, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_C);
    cudaFree(d_A);

    const float tol = 1e-4f;
    if (fabsf(result - expected) > tol) {
        printf("\n  Expected %.4f got %.4f", expected, result);
        return 1;
    }
    return 0;
}

/* =========================================================================
 * main
 * =========================================================================*/
int main(void)
{
    curfpHandle_t handle;
    if (curfpCreate(&handle) != CURFP_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create curfp handle\n");
        return 1;
    }

    int failures = 0;

    const curfpOperation_t transr_vals[] = { CURFP_OP_N, CURFP_OP_T };
    const curfpFillMode_t  uplo_vals[]   = { CURFP_FILL_MODE_LOWER, CURFP_FILL_MODE_UPPER };
    const char *transr_names[] = { "N", "T" };
    const char *uplo_names[]   = { "L", "U" };
    const int   n_vals[]       = { 5, 6 };

    /* --- Test 1 & 2: parameter corner cases for all 8 RFP variants --- */
    for (int ti = 0; ti < 2; ti++) {
        for (int ui = 0; ui < 2; ui++) {
            for (int ni = 0; ni < 2; ni++) {
                int n = n_vals[ni];

                printf("ssfrk n=%d transr=%s uplo=%s beta=0  ... ",
                       n, transr_names[ti], uplo_names[ui]);
                int r = test_beta_zero(handle, n, transr_vals[ti], uplo_vals[ui]);
                printf("%s\n", r == 0 ? "PASS" : "FAIL");
                failures += r;

                printf("ssfrk n=%d transr=%s uplo=%s alpha=0 ... ",
                       n, transr_names[ti], uplo_names[ui]);
                r = test_alpha_zero_beta_one(handle, n, transr_vals[ti], uplo_vals[ui]);
                printf("%s\n", r == 0 ? "PASS" : "FAIL");
                failures += r;
            }
        }
    }

    /* --- Test 3: n=2, TRANSR=N, UPLO=L numerical --- */
    printf("ssfrk n=2 transr=N uplo=L rank-1 numerical ... ");
    {
        int r = test_n2_NL_rankk(handle);
        printf("%s\n", r == 0 ? "PASS" : "FAIL");
        failures += r;
    }

    /* --- Test 4: n=1 for all 4 variants --- */
    for (int ti = 0; ti < 2; ti++) {
        for (int ui = 0; ui < 2; ui++) {
            printf("ssfrk n=1 transr=%s uplo=%s rank-1 ... ",
                   transr_names[ti], uplo_names[ui]);
            int r = test_n1_rankk(handle, transr_vals[ti], uplo_vals[ui]);
            printf("%s\n", r == 0 ? "PASS" : "FAIL");
            failures += r;
        }
    }

    curfpDestroy(handle);
    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "FAILED", failures);
    return failures != 0 ? 1 : 0;
}
