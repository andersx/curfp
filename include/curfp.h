#ifndef CURFP_H
#define CURFP_H

#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Status codes
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_STATUS_SUCCESS          = 0,
    CURFP_STATUS_NOT_INITIALIZED  = 1,
    CURFP_STATUS_ALLOC_FAILED     = 2,
    CURFP_STATUS_INVALID_VALUE    = 3,
    CURFP_STATUS_EXECUTION_FAILED = 4
} curfpStatus_t;

/* ---------------------------------------------------------------------------
 * Operation type (controls TRANSR and TRANS parameters)
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_OP_N = 0,  /* Non-transpose */
    CURFP_OP_T = 1   /* Transpose     */
} curfpOperation_t;

/* ---------------------------------------------------------------------------
 * Fill mode (upper or lower triangle)
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_FILL_MODE_LOWER = 0,
    CURFP_FILL_MODE_UPPER = 1
} curfpFillMode_t;

/* ---------------------------------------------------------------------------
 * Opaque handle
 * ---------------------------------------------------------------------------*/
typedef struct curfpContext *curfpHandle_t;

/* ---------------------------------------------------------------------------
 * Handle management
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpCreate(curfpHandle_t *handle);
curfpStatus_t curfpDestroy(curfpHandle_t handle);
curfpStatus_t curfpSetStream(curfpHandle_t handle, cudaStream_t stream);
curfpStatus_t curfpGetStream(curfpHandle_t handle, cudaStream_t *stream);

/* ---------------------------------------------------------------------------
 * curfpSsfrk — Symmetric Rank-K update in RFP format (single precision)
 *
 * Computes:
 *   trans == CURFP_OP_N:  C := alpha * A * A^T + beta * C   (A is N x K)
 *   trans == CURFP_OP_T:  C := alpha * A^T * A + beta * C   (A is K x N)
 *
 * C is an N x N symmetric matrix stored in RFP format (N*(N+1)/2 floats on
 * device).  transr selects the RFP storage variant (normal or transposed).
 * alpha and beta are host pointers; A and C are device pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfrk(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    curfpOperation_t trans,    /* operation on A: CURFP_OP_N or CURFP_OP_T       */
    int              n,        /* order of symmetric matrix C                    */
    int              k,        /* number of columns (trans=N) or rows (trans=T)  */
    const float     *alpha,    /* host pointer                                   */
    const float     *A,        /* device pointer, leading dimension lda          */
    int              lda,
    const float     *beta,     /* host pointer                                   */
    float           *C         /* device pointer, RFP format, N*(N+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSpftrf — Cholesky factorization in RFP format (single precision)
 *
 * Computes:
 *   uplo == CURFP_FILL_MODE_LOWER:  A = L * L^T
 *   uplo == CURFP_FILL_MODE_UPPER:  A = U^T * U
 *
 * A is an N x N symmetric positive definite matrix stored in RFP format
 * (N*(N+1)/2 floats on device).  On exit A is overwritten with the factor.
 * info is a host pointer: 0 = success, >0 = leading minor of order info is
 * not positive definite.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSpftrf(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of matrix A                              */
    float           *A,        /* device pointer, RFP format, N*(N+1)/2 floats  */
    int             *info      /* host pointer: 0 = success                     */
);

#ifdef __cplusplus
}
#endif

#endif /* CURFP_H */
