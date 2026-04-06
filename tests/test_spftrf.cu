/*
 * test_spftrf.cu — Correctness tests for curfpSpftrf
 *
 * Tests:
 *   1. n=1, all 4 transr/uplo variants: single element → sqrt of that element.
 *   2. n=2, TRANSR=N, UPLO=L (Case 5): hand-verified numerical.
 *   3. n=3, TRANSR=N, UPLO=L (Case 1): full Cholesky factor verification.
 *   4. Smoke: all 8 variants return CURFP_STATUS_SUCCESS for n=1.
 *
 * RFP layout derivation (verified against pointer offsets in curfp_spftrf.cpp):
 *
 * n=2, TRANSR=N, UPLO=L (Case 5, n even, k=1, lda_rfp=n+1=3):
 *   spftrf calls: SPOTRF on A+1, STRSM writes A+2, SSYRK+SPOTRF on A+0
 *   ARF[0] = A(1,1)   ARF[1] = A(0,0)   ARF[2] = A(1,0)
 *
 * n=3, TRANSR=N, UPLO=L (Case 1, n odd, n1=2, n2=1, lda_rfp=n=3):
 *   spftrf calls: SPOTRF on A+0 (n1=2 block, lda=3), STRSM writes A+n1=A+2,
 *                SSYRK+SPOTRF on A+n=A+3
 *   A11(0,0) at A[0], A11(1,0) at A[1], A11(1,1) at A[4]
 *   A21(0,0) at A[2] (= A(2,0)), A21(0,1) at A[5] (= A(2,1))
 *   A22(0,0) at A[3] (= A(2,2))
 *   ARF = { A(0,0), A(1,0), A(2,0), A(2,2), A(1,1), A(2,1) }
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "curfp.h"

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
 * Test 1: n=1 — all 4 transr/uplo variants.
 * =========================================================================*/
static int test_n1(curfpHandle_t handle,
                   curfpOperation_t transr, curfpFillMode_t uplo)
{
    const float val      = 9.0f;
    const float expected = sqrtf(val);

    float *d_ARF;
    CHECK_CUDA(cudaMalloc(&d_ARF, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ARF, &val, sizeof(float), cudaMemcpyHostToDevice));

    int info = -1;
    CHECK_CURFP(curfpSpftrf(handle, transr, uplo, 1, d_ARF, &info));

    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_ARF, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_ARF);

    if (info != 0) { printf("\n  info=%d (expected 0)", info); return 1; }

    if (fabsf(result - expected) > 1e-5f) {
        printf("\n  Expected %.6f got %.6f", expected, result);
        return 1;
    }
    return 0;
}

/* =========================================================================
 * Test 2: n=2, TRANSR=N, UPLO=L (Case 5)
 *
 * Input A = [[4, 2], [2, 3]]
 * RFP encoding: ARF = {A(1,1), A(0,0), A(1,0)} = {3, 4, 2}
 *
 * Cholesky lower factor L:
 *   L(0,0) = sqrt(4) = 2
 *   L(1,0) = 2 / 2 = 1
 *   L(1,1) = sqrt(3 - 1^2) = sqrt(2)
 *
 * RFP output (same encoding): {L(1,1), L(0,0), L(1,0)} = {sqrt(2), 2, 1}
 * =========================================================================*/
static int test_n2_NL(curfpHandle_t handle)
{
    const float a00 = 4.0f, a10 = 2.0f, a11 = 3.0f;
    float h_ARF[3] = { a11, a00, a10 };

    const float exp_l00 = sqrtf(a00);
    const float exp_l10 = a10 / exp_l00;
    const float exp_l11 = sqrtf(a11 - exp_l10 * exp_l10);

    float *d_ARF;
    CHECK_CUDA(cudaMalloc(&d_ARF, 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ARF, h_ARF, 3 * sizeof(float), cudaMemcpyHostToDevice));

    int info = -1;
    CHECK_CURFP(curfpSpftrf(handle, CURFP_OP_N, CURFP_FILL_MODE_LOWER,
                             2, d_ARF, &info));

    float h_result[3];
    CHECK_CUDA(cudaMemcpy(h_result, d_ARF, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_ARF);

    if (info != 0) { printf("\n  info=%d (expected 0)", info); return 1; }

    /* Output encoding: h_result[1]=L(0,0), h_result[2]=L(1,0), h_result[0]=L(1,1) */
    const float tol = 1e-4f;
    int ok = (fabsf(h_result[1] - exp_l00) < tol) &&
             (fabsf(h_result[2] - exp_l10) < tol) &&
             (fabsf(h_result[0] - exp_l11) < tol);
    if (!ok) {
        printf("\n  Expected: L(0,0)=%.5f L(1,0)=%.5f L(1,1)=%.5f",
               exp_l00, exp_l10, exp_l11);
        printf("\n  Got:      L(0,0)=%.5f L(1,0)=%.5f L(1,1)=%.5f",
               h_result[1], h_result[2], h_result[0]);
    }
    return ok ? 0 : 1;
}

/* =========================================================================
 * Test 3: n=3, TRANSR=N, UPLO=L (Case 1)
 *
 * n=3 odd, n1=2, n2=1, lda_rfp=3.
 * Pointer layout (from spftrf Case 1: SPOTRF on A+0 (2x2,lda=3),
 *                STRSM output at A+2 (1x2,lda=3), SPOTRF on A+3 (1x1)):
 *   A11 block (2x2 lower, lda=3): A[0]=A(0,0), A[1]=A(1,0), A[4]=A(1,1)
 *   A21 block (1x2, lda=3):       A[2]=A(2,0), A[5]=A(2,1)
 *   A22 block (1x1, lda=3):       A[3]=A(2,2)
 *
 *   RFP input = { A(0,0), A(1,0), A(2,0), A(2,2), A(1,1), A(2,1) }
 *             = {  5,      1,      2,      6,      4,      0      }
 *
 * Cholesky of A = [[5,1,2],[1,4,0],[2,0,6]]:
 *   L(0,0) = sqrt(5)
 *   L(1,0) = 1/sqrt(5)
 *   L(2,0) = 2/sqrt(5)
 *   L(1,1) = sqrt(4 - 1/5) = sqrt(19/5)
 *   L(2,1) = (0 - (2/sqrt(5))*(1/sqrt(5))) / sqrt(19/5)
 *   L(2,2) = sqrt(6 - L(2,0)^2 - L(2,1)^2)
 *
 * RFP output = { L(0,0), L(1,0), L(2,0), L(2,2), L(1,1), L(2,1) }
 * =========================================================================*/
static int test_n3_NL(curfpHandle_t handle)
{
    /* Dense A, col-major, lda=3: A[j*3+i] = A(i,j) */
    const float a00=5.0f, a10=1.0f, a20=2.0f;
    const float a11=4.0f, a21=0.0f;
    const float a22=6.0f;

    /* RFP input (Case 1 layout): { A(0,0), A(1,0), A(2,0), A(2,2), A(1,1), A(2,1) } */
    float h_ARF[6] = { a00, a10, a20, a22, a11, a21 };

    /* Reference Cholesky lower factor */
    const float exp_l00 = sqrtf(a00);
    const float exp_l10 = a10 / exp_l00;
    const float exp_l20 = a20 / exp_l00;
    const float exp_l11 = sqrtf(a11 - exp_l10 * exp_l10);
    const float exp_l21 = (a21 - exp_l20 * exp_l10) / exp_l11;
    const float exp_l22 = sqrtf(a22 - exp_l20 * exp_l20 - exp_l21 * exp_l21);

    float *d_ARF;
    CHECK_CUDA(cudaMalloc(&d_ARF, 6 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ARF, h_ARF, 6 * sizeof(float), cudaMemcpyHostToDevice));

    int info = -1;
    CHECK_CURFP(curfpSpftrf(handle, CURFP_OP_N, CURFP_FILL_MODE_LOWER,
                             3, d_ARF, &info));

    float h_result[6];
    CHECK_CUDA(cudaMemcpy(h_result, d_ARF, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_ARF);

    if (info != 0) { printf("\n  info=%d (expected 0)", info); return 1; }

    /* Output uses same layout: { L(0,0), L(1,0), L(2,0), L(2,2), L(1,1), L(2,1) } */
    const float tol = 1e-4f;
    int ok = (fabsf(h_result[0] - exp_l00) < tol) &&
             (fabsf(h_result[1] - exp_l10) < tol) &&
             (fabsf(h_result[2] - exp_l20) < tol) &&
             (fabsf(h_result[4] - exp_l11) < tol) &&
             (fabsf(h_result[5] - exp_l21) < tol) &&
             (fabsf(h_result[3] - exp_l22) < tol);
    if (!ok) {
        printf("\n  Expected: L00=%.5f L10=%.5f L20=%.5f L11=%.5f L21=%.5f L22=%.5f",
               exp_l00, exp_l10, exp_l20, exp_l11, exp_l21, exp_l22);
        printf("\n  Got:      L00=%.5f L10=%.5f L20=%.5f L11=%.5f L21=%.5f L22=%.5f",
               h_result[0], h_result[1], h_result[2], h_result[4], h_result[5], h_result[3]);
    }
    return ok ? 0 : 1;
}

/* =========================================================================
 * Test 4: smoke — all 8 RFP variants, n=1 (always valid SPD input).
 * =========================================================================*/
static int test_smoke_n1(curfpHandle_t handle,
                         curfpOperation_t transr, curfpFillMode_t uplo)
{
    const float val = 4.0f;
    float *d_ARF;
    CHECK_CUDA(cudaMalloc(&d_ARF, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ARF, &val, sizeof(float), cudaMemcpyHostToDevice));

    int info = -1;
    curfpStatus_t st = curfpSpftrf(handle, transr, uplo, 1, d_ARF, &info);
    cudaFree(d_ARF);

    if (st != CURFP_STATUS_SUCCESS) {
        printf("\n  status=%d", (int)st);
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

    /* --- Test 1: n=1 for all 4 transr/uplo variants --- */
    for (int ti = 0; ti < 2; ti++) {
        for (int ui = 0; ui < 2; ui++) {
            printf("spftrf n=1 transr=%s uplo=%s ... ",
                   transr_names[ti], uplo_names[ui]);
            int r = test_n1(handle, transr_vals[ti], uplo_vals[ui]);
            printf("%s\n", r == 0 ? "PASS" : "FAIL");
            failures += r;
        }
    }

    /* --- Test 2: n=2, TRANSR=N, UPLO=L numerical --- */
    printf("spftrf n=2 transr=N uplo=L numerical ... ");
    {
        int r = test_n2_NL(handle);
        printf("%s\n", r == 0 ? "PASS" : "FAIL");
        failures += r;
    }

    /* --- Test 3: n=3, TRANSR=N, UPLO=L full factor verification --- */
    printf("spftrf n=3 transr=N uplo=L full L verification ... ");
    {
        int r = test_n3_NL(handle);
        printf("%s\n", r == 0 ? "PASS" : "FAIL");
        failures += r;
    }

    /* --- Test 4: smoke tests for all 8 variants --- */
    for (int ti = 0; ti < 2; ti++) {
        for (int ui = 0; ui < 2; ui++) {
            printf("spftrf smoke n=1 transr=%s uplo=%s ... ",
                   transr_names[ti], uplo_names[ui]);
            int r = test_smoke_n1(handle, transr_vals[ti], uplo_vals[ui]);
            printf("%s\n", r == 0 ? "PASS" : "FAIL");
            failures += r;
        }
    }

    curfpDestroy(handle);
    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "FAILED", failures);
    return failures != 0 ? 1 : 0;
}
