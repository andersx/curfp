/*
 * curfpSpfcon — Reciprocal condition number estimator for an SPD matrix
 *               from its RFP Cholesky factor (single precision).
 *
 * Computes:
 *   rcond = 1 / ( ||A^{-1}||_1 * anorm )
 *
 * where anorm = ||A||_1 of the original (pre-factorization) matrix, supplied
 * by the caller.  Analogous to LAPACK SPOCON but operates directly on the
 * RFP Cholesky factor without unpacking.
 *
 * Algorithm: port of LAPACK SPOCON/SLACN2 (Hager–Higham iterative 1-norm
 * estimator), max ITMAX=5 iterations.  The iterate vector d_x stays on GPU
 * throughout — no host↔device vector copies.
 *
 * GPU operations used per iteration:
 *   cublasSasum   — 1-norm, scalar returned to host
 *   cublasIsamax  — argmax index, scalar returned to host
 *   cublasSdot    — sign consistency check, scalar returned to host
 *   curfpSpftrs   — in-place triangular solve on d_x (one RHS)
 *   4 tiny kernels — fill-uniform, sign-and-save, set-e_j, altsgn-fill
 *
 * For SPD matrices A^{-T} = A^{-1}, so SLACN2 kase=1 and kase=2 both
 * reduce to the same curfpSpftrs call.
 */

#include <math.h>
#include "curfp_internal.h"

/* Forward declaration of curfpSpftrs (defined in curfp_spftrs.cpp) */
extern "C"
curfpStatus_t curfpSpftrs(curfpHandle_t    handle,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           int              nrhs,
                           const float     *A,
                           float           *B,
                           int              ldb);

/* -------------------------------------------------------------------------
 * Device kernels
 * ------------------------------------------------------------------------- */

/* x[i] = val for all i */
__global__ static void k_fill_uniform(float *x, int n, float val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}

/* x[i] = sign(x[i]);  isgn[i] = sign(x[i]) as ±1.0f */
__global__ static void k_sign_and_save(float *x, float *isgn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = (x[i] >= 0.0f) ? 1.0f : -1.0f;
        x[i]    = s;
        isgn[i] = s;
    }
}

/* x = e_j: x[j-1] = 1.0, all others 0.0  (j is 1-based) */
__global__ static void k_set_ej(float *x, int n, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (i == j - 1) ? 1.0f : 0.0f;
}

/* x[i] = (±1) * (1 + i/(n-1)), alternating sign from +1.
 * Special case n==1: x[0] = 1.0. */
__global__ static void k_altsgn_fill(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sgn   = (i % 2 == 0) ? 1.0f : -1.0f;
        float scale = (n > 1) ? (1.0f + (float)i / (float)(n - 1)) : 1.0f;
        x[i] = sgn * scale;
    }
}

#define LAUNCH1D(kern, stream, n, ...) \
    kern<<<((n) + 255) / 256, 256, 0, (stream)>>>(__VA_ARGS__)

/* =========================================================================
 * Public entry point
 * ========================================================================= */
extern "C"
curfpStatus_t curfpSpfcon(curfpHandle_t    handle,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           const float     *arf,
                           float            anorm,
                           float           *rcond)
{
    CURFP_CHECK_HANDLE(handle);
    if (!rcond) return CURFP_STATUS_INVALID_VALUE;

    *rcond = 0.0f;
    if (n == 0 || anorm == 0.0f) return CURFP_STATUS_SUCCESS;
    if (n < 0)                   return CURFP_STATUS_INVALID_VALUE;

    cublasHandle_t cb = handle->cublas;

    /* Declare all locals before any goto to avoid jumping over initializers */
    cudaStream_t        stream;
    cublasPointerMode_t old_mode;
    float *d_x    = NULL;
    float *d_isgn = NULL;
    curfpStatus_t status = CURFP_STATUS_SUCCESS;

    /* SLACN2 scalar state */
    float est   = 0.0f;
    int   j     = 0;    /* 1-based argmax index */
    int   iter  = 0;
    int   jump  = 0;    /* 0 = initial (not part of LAPACK SLACN2 numbering) */
    int   done  = 0;
    const int ITMAX = 5;

    /* Get stream and save/set pointer mode */
    if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS) {
        return CURFP_STATUS_EXECUTION_FAILED;
    }
    if (cublasGetPointerMode(cb, &old_mode) != CUBLAS_STATUS_SUCCESS) {
        return CURFP_STATUS_EXECUTION_FAILED;
    }
    if (cublasSetPointerMode(cb, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) {
        return CURFP_STATUS_EXECUTION_FAILED;
    }

    /* Allocate GPU work vectors (stream-ordered: avoids device-wide sync) */
    if (cudaMallocAsync((void **)&d_x,    (size_t)n * sizeof(float), stream) != cudaSuccess ||
        cudaMallocAsync((void **)&d_isgn, (size_t)n * sizeof(float), stream) != cudaSuccess) {
        status = CURFP_STATUS_ALLOC_FAILED;
        goto cleanup;
    }

/* Local macros: on error, set status and jump to cleanup */
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
#define CHK_ST(expr) \
    do { curfpStatus_t _s = (expr); \
         if (_s != CURFP_STATUS_SUCCESS) { status = _s; goto cleanup; } \
    } while (0)

    /* ---- SLACN2 main loop -------------------------------------------- */
    while (!done) {
        float sasum_val = 0.0f;
        int   imax_val  = 0;
        float dot_val   = 0.0f;
        int   do_solve  = 0;

        switch (jump) {

        /* ---- Initial: fill d_x = 1/n, do first A^{-1} apply --------- */
        case 0:
            LAUNCH1D(k_fill_uniform, stream, n, d_x, n, 1.0f / (float)n);
            jump     = 1;
            do_solve = 1;
            break;

        /* ---- jump=1: after A^{-1} * (1/n * 1) ------------------------ */
        case 1:
            CHK_CB(cublasSasum(cb, n, d_x, 1, &sasum_val));
            if (n == 1) { est = sasum_val; done = 1; break; }
            est = sasum_val;
            /* x = sign(x), save to isgn */
            LAUNCH1D(k_sign_and_save, stream, n, d_x, d_isgn, n);
            jump     = 2;
            do_solve = 1;   /* kase=2 = kase=1 for SPD */
            break;

        /* ---- jump=2: after A^{-1} * sign(x) -------------------------- */
        case 2:
            CHK_CB(cublasIsamax(cb, n, d_x, 1, &imax_val));
            j    = imax_val;
            iter = 2;
            LAUNCH1D(k_set_ej, stream, n, d_x, n, j);
            jump     = 3;
            do_solve = 1;   /* kase=1: apply A^{-1} to e_j */
            break;

        /* ---- jump=3: after A^{-1} * e_j ------------------------------ */
        case 3:
            CHK_CB(cublasSasum(cb, n, d_x, 1, &sasum_val));
            CHK_CB(cublasSdot(cb, n, d_isgn, 1, d_x, 1, &dot_val));
            est = sasum_val;
            if (dot_val == sasum_val) {
                /* All signs unchanged → converged, proceed to final phase */
                LAUNCH1D(k_altsgn_fill, stream, n, d_x, n);
                jump     = 5;
                do_solve = 1;
                break;
            }
            /* Signs changed: update d_x and d_isgn, do another A^{-1}*sign(x) */
            LAUNCH1D(k_sign_and_save, stream, n, d_x, d_isgn, n);
            jump     = 4;
            do_solve = 1;   /* kase=2 = kase=1 for SPD */
            break;

        /* ---- jump=4: after A^{-1} * sign(x) in main loop ------------- */
        case 4: {
            CHK_CB(cublasIsamax(cb, n, d_x, 1, &imax_val));
            int   j_new   = imax_val;
            float xv_old  = 0.0f;
            float xv_new  = 0.0f;
            /* Copy two scalar values from device to compare convergence */
            CHK_CU(cudaMemcpyAsync(&xv_old, d_x + (j     - 1), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
            CHK_CU(cudaMemcpyAsync(&xv_new, d_x + (j_new - 1), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
            CHK_CU(cudaStreamSynchronize(stream));
            if (fabsf(xv_old) != fabsf(xv_new) && iter < ITMAX) {
                j = j_new;
                iter++;
                LAUNCH1D(k_set_ej, stream, n, d_x, n, j);
                jump     = 3;
                do_solve = 1;
            } else {
                /* Converged (or max iterations reached) → final phase */
                LAUNCH1D(k_altsgn_fill, stream, n, d_x, n);
                jump     = 5;
                do_solve = 1;
            }
            break;
        }

        /* ---- jump=5: after A^{-1} * altsgn_fill ---------------------- */
        case 5: {
            CHK_CB(cublasSasum(cb, n, d_x, 1, &sasum_val));
            float temp = 2.0f * sasum_val / (3.0f * (float)n);
            if (temp > est) est = temp;
            done = 1;
            break;
        }

        default:
            status = CURFP_STATUS_EXECUTION_FAILED;
            done   = 1;
            break;
        }

        if (!done && do_solve) {
            /* Apply A^{-1} in-place on d_x (one column, column-major trivial) */
            CHK_ST(curfpSpftrs(handle, transr, uplo, n, 1, arf, d_x, n));
        }
    }

#undef CHK_CB
#undef CHK_CU
#undef CHK_ST

cleanup:
    cublasSetPointerMode(cb, old_mode);
    if (d_x)    cudaFreeAsync(d_x,    stream);
    if (d_isgn) cudaFreeAsync(d_isgn, stream);

    if (status == CURFP_STATUS_SUCCESS && est > 0.0f)
        *rcond = (1.0f / est) / anorm;

    return status;
}
