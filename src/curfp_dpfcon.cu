/*
 * curfpDpfcon — Reciprocal condition number estimator for an SPD matrix
 *               from its RFP Cholesky factor (double precision).
 *
 * Double-precision copy of curfp_spfcon.cu: float -> double,
 * cublasSasum -> cublasDasum, cublasIsamax -> cublasIdamax,
 * cublasSdot -> cublasDdot, fabsf -> fabs, float literals -> double.
 * Forward declaration references curfpDpftrs (not curfpSpftrs).
 */

#include <math.h>
#include "curfp_internal.h"

extern "C"
curfpStatus_t curfpDpftrs(curfpHandle_t    handle,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           int              nrhs,
                           const double    *A,
                           double          *B,
                           int              ldb);

__global__ static void k_fill_uniform_d(double *x, int n, double val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}

__global__ static void k_sign_and_save_d(double *x, double *isgn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = (x[i] >= 0.0) ? 1.0 : -1.0;
        x[i]    = s;
        isgn[i] = s;
    }
}

__global__ static void k_set_ej_d(double *x, int n, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (i == j - 1) ? 1.0 : 0.0;
}

__global__ static void k_altsgn_fill_d(double *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double sgn   = (i % 2 == 0) ? 1.0 : -1.0;
        double scale = (n > 1) ? (1.0 + (double)i / (double)(n - 1)) : 1.0;
        x[i] = sgn * scale;
    }
}

#define LAUNCH1D(kern, stream, n, ...) \
    kern<<<((n) + 255) / 256, 256, 0, (stream)>>>(__VA_ARGS__)

extern "C"
curfpStatus_t curfpDpfcon(curfpHandle_t    handle,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           const double    *arf,
                           double           anorm,
                           double          *rcond)
{
    CURFP_CHECK_HANDLE(handle);
    if (!rcond) return CURFP_STATUS_INVALID_VALUE;

    *rcond = 0.0;
    if (n == 0 || anorm == 0.0) return CURFP_STATUS_SUCCESS;
    if (n < 0)                  return CURFP_STATUS_INVALID_VALUE;

    cublasHandle_t cb = handle->cublas;

    cudaStream_t        stream;
    cublasPointerMode_t old_mode;
    double *d_x    = NULL;
    double *d_isgn = NULL;
    curfpStatus_t status = CURFP_STATUS_SUCCESS;

    double est   = 0.0;
    int    j     = 0;
    int    iter  = 0;
    int    jump  = 0;
    int    done  = 0;
    const int ITMAX = 5;

    if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;
    if (cublasGetPointerMode(cb, &old_mode) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;
    if (cublasSetPointerMode(cb, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS)
        return CURFP_STATUS_EXECUTION_FAILED;

    if (cudaMallocAsync((void **)&d_x,    (size_t)n * sizeof(double), stream) != cudaSuccess ||
        cudaMallocAsync((void **)&d_isgn, (size_t)n * sizeof(double), stream) != cudaSuccess) {
        status = CURFP_STATUS_ALLOC_FAILED;
        goto cleanup;
    }

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

    while (!done) {
        double dasum_val = 0.0;
        int    imax_val  = 0;
        double dot_val   = 0.0;
        int    do_solve  = 0;

        switch (jump) {

        case 0:
            LAUNCH1D(k_fill_uniform_d, stream, n, d_x, n, 1.0 / (double)n);
            jump     = 1;
            do_solve = 1;
            break;

        case 1:
            CHK_CB(cublasDasum(cb, n, d_x, 1, &dasum_val));
            if (n == 1) { est = dasum_val; done = 1; break; }
            est = dasum_val;
            LAUNCH1D(k_sign_and_save_d, stream, n, d_x, d_isgn, n);
            jump     = 2;
            do_solve = 1;
            break;

        case 2:
            CHK_CB(cublasIdamax(cb, n, d_x, 1, &imax_val));
            j    = imax_val;
            iter = 2;
            LAUNCH1D(k_set_ej_d, stream, n, d_x, n, j);
            jump     = 3;
            do_solve = 1;
            break;

        case 3:
            CHK_CB(cublasDasum(cb, n, d_x, 1, &dasum_val));
            CHK_CB(cublasDdot(cb, n, d_isgn, 1, d_x, 1, &dot_val));
            est = dasum_val;
            if (dot_val == dasum_val) {
                LAUNCH1D(k_altsgn_fill_d, stream, n, d_x, n);
                jump     = 5;
                do_solve = 1;
                break;
            }
            LAUNCH1D(k_sign_and_save_d, stream, n, d_x, d_isgn, n);
            jump     = 4;
            do_solve = 1;
            break;

        case 4: {
            CHK_CB(cublasIdamax(cb, n, d_x, 1, &imax_val));
            int    j_new   = imax_val;
            double xv_old  = 0.0;
            double xv_new  = 0.0;
            CHK_CU(cudaMemcpyAsync(&xv_old, d_x + (j     - 1), sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
            CHK_CU(cudaMemcpyAsync(&xv_new, d_x + (j_new - 1), sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
            CHK_CU(cudaStreamSynchronize(stream));
            if (fabs(xv_old) != fabs(xv_new) && iter < ITMAX) {
                j = j_new;
                iter++;
                LAUNCH1D(k_set_ej_d, stream, n, d_x, n, j);
                jump     = 3;
                do_solve = 1;
            } else {
                LAUNCH1D(k_altsgn_fill_d, stream, n, d_x, n);
                jump     = 5;
                do_solve = 1;
            }
            break;
        }

        case 5: {
            CHK_CB(cublasDasum(cb, n, d_x, 1, &dasum_val));
            double temp = 2.0 * dasum_val / (3.0 * (double)n);
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
            CHK_ST(curfpDpftrs(handle, transr, uplo, n, 1, arf, d_x, n));
        }
    }

#undef CHK_CB
#undef CHK_CU
#undef CHK_ST

cleanup:
    cublasSetPointerMode(cb, old_mode);
    if (d_x)    cudaFreeAsync(d_x,    stream);
    if (d_isgn) cudaFreeAsync(d_isgn, stream);

    if (status == CURFP_STATUS_SUCCESS && est > 0.0)
        *rcond = (1.0 / est) / anorm;

    return status;
}
