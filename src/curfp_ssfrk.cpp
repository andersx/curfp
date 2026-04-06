/*
 * curfpSsfrk — Symmetric Rank-K update in RFP format (single precision)
 *
 * Direct translation of LAPACK ssfrk.f to CUDA using cuBLAS for the
 * underlying BLAS calls.  The RFP array C is partitioned into blocks;
 * pointer offsets follow the LAPACK reference source exactly, converting
 * from 1-based Fortran indexing (C(*)) to 0-based C pointers.
 *
 * There are 8 RFP storage variants:
 *   N parity (odd/even) × transr (N/T) × uplo (L/U)
 * Each variant computes 2× cublasSsyrk + 1× cublasSgemm on the sub-blocks.
 */

#include "curfp_internal.h"

curfpStatus_t curfpSsfrk(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpOperation_t trans,
    int              n,
    int              k,
    const float     *alpha,
    const float     *A,
    int              lda,
    const float     *beta,
    float           *C)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || k < 0 || lda < 1) return CURFP_STATUS_INVALID_VALUE;
    if (!alpha || !beta)            return CURFP_STATUS_INVALID_VALUE;
    if (n == 0)                     return CURFP_STATUS_SUCCESS;

    const float zero = 0.0f, one = 1.0f;

    /* Quick return when result is trivially beta*C */
    if ((*alpha == 0.0f || k == 0) && *beta == 1.0f)
        return CURFP_STATUS_SUCCESS;

    /* If alpha==0 and beta==0 just zero the RFP array */
    if (*alpha == 0.0f && *beta == 0.0f) {
        /* The RFP array has n*(n+1)/2 elements */
        long ntotal = (long)n * (n + 1) / 2;
        CURFP_CHECK_CUDA(cudaMemset(C, 0, ntotal * sizeof(float)));
        return CURFP_STATUS_SUCCESS;
    }

    cublasHandle_t cb = handle->cublas;

    const int notrans     = (trans  == CURFP_OP_N);
    const int normaltransr = (transr == CURFP_OP_N);
    const int lower       = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd      = (n % 2 != 0);

    /* Sub-block dimensions */
    int n1, n2;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        n1 = n / 2;
        n2 = n1;      /* only NK is used for even N */
    }

    /* cuBLAS enums for the operation on A */
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasOperation_t opT = CUBLAS_OP_T;

    /* Pointer to second block of A:
     *   notrans: row offset  → A + n1 (rows 1..n1 of A in col-major = column shift)
     *   trans:   col offset  → A + n1*lda
     * For the odd/N/U case the second block starts at row n2 of A (A(N2,1) in
     * Fortran), i.e. offset n2-1 or n2-1 * lda for the transpose case.
     */

    /* =====================================================================
     * 8 RFP cases
     * ===================================================================== */

    if (nisodd) {
        /* ------------------------------------------------------------------ */
        /* Odd N                                                               */
        /* ------------------------------------------------------------------ */
        if (normaltransr) {
            /* ldc for sub-blocks = N */
            if (lower) {
                /* Case 1: odd, TRANSR=N, UPLO=L
                 *   SSYRK('L', trans, n1, k, alpha, A1, lda, beta, C+0,  n)
                 *   SSYRK('U', trans, n2, k, alpha, A2, lda, beta, C+n,  n)
                 *   SGEMM                       ...         beta, C+n1, n)
                 * Fortran: C(1)=C+0, C(N+1)=C+n, C(N1+1)=C+n1  (0-based)
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + n1 : A + (long)n1 * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    n1, k, alpha, A1, lda, beta, C + 0, n));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    n2, k, alpha, A2, lda, beta, C + n, n));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    n2, n1, k,
                    alpha, A2, lda, A1, lda,
                    beta, C + n1, n));

            } else {
                /* Case 2: odd, TRANSR=N, UPLO=U
                 * Fortran: C(N2+1)=C+n2, C(N1+1)=C+n1, C(1)=C+0
                 * A1=A(1,1), A2=A(N2,1) i.e. row n2 (0-based row n2-1)
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + (n2 - 1)
                                          : A + (long)(n2 - 1) * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    n1, k, alpha, A1, lda, beta, C + n2, n));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    n2, k, alpha, A2, lda, beta, C + n1, n));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    n1, n2, k,
                    alpha, A1, lda, A2, lda,
                    beta, C + 0, n));
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 3: odd, TRANSR=T, UPLO=L
                 * ldc = n1
                 * Fortran: C(1)=C+0, C(2)=C+1, C(N1*N1+1)=C+n1*n1
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + n1 : A + (long)n1 * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    n1, k, alpha, A1, lda, beta, C + 0, n1));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    n2, k, alpha, A2, lda, beta, C + 1, n1));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    n1, n2, k,
                    alpha, A1, lda, A2, lda,
                    beta, C + (long)n1 * n1, n1));

            } else {
                /* Case 4: odd, TRANSR=T, UPLO=U
                 * ldc = n2
                 * Fortran: C(N2*N2+1)=C+n2*n2, C(N1*N2+1)=C+n1*n2, C(1)=C+0
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + n1 : A + (long)n1 * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    n1, k, alpha, A1, lda, beta, C + (long)n2 * n2, n2));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    n2, k, alpha, A2, lda, beta, C + (long)n1 * n2, n2));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    n2, n1, k,
                    alpha, A2, lda, A1, lda,
                    beta, C + 0, n2));
            }
        }
    } else {
        /* ------------------------------------------------------------------ */
        /* Even N: NK = N/2                                                    */
        /* ------------------------------------------------------------------ */
        const int nk = n1;  /* == n2 == n/2 */

        if (normaltransr) {
            /* ldc for sub-blocks = N+1 */
            if (lower) {
                /* Case 5: even, TRANSR=N, UPLO=L
                 * Fortran: C(2)=C+1, C(1)=C+0, C(NK+2)=C+nk+1
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + nk : A + (long)nk * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    nk, k, alpha, A1, lda, beta, C + 1, n + 1));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    nk, k, alpha, A2, lda, beta, C + 0, n + 1));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    nk, nk, k,
                    alpha, A2, lda, A1, lda,
                    beta, C + nk + 1, n + 1));

            } else {
                /* Case 6: even, TRANSR=N, UPLO=U
                 * Fortran: C(NK+2)=C+nk+1, C(NK+1)=C+nk, C(1)=C+0
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + nk : A + (long)nk * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    nk, k, alpha, A1, lda, beta, C + nk + 1, n + 1));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    nk, k, alpha, A2, lda, beta, C + nk, n + 1));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    nk, nk, k,
                    alpha, A1, lda, A2, lda,
                    beta, C + 0, n + 1));
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 7: even, TRANSR=T, UPLO=L
                 * ldc = nk
                 * Fortran: C(NK+1)=C+nk, C(1)=C+0, C((NK+1)*NK+1)=C+(nk+1)*nk
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + nk : A + (long)nk * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    nk, k, alpha, A1, lda, beta, C + nk, nk));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    nk, k, alpha, A2, lda, beta, C + 0, nk));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    nk, nk, k,
                    alpha, A1, lda, A2, lda,
                    beta, C + (long)(nk + 1) * nk, nk));

            } else {
                /* Case 8: even, TRANSR=T, UPLO=U
                 * ldc = nk
                 * Fortran: C(NK*(NK+1)+1)=C+nk*(nk+1), C(NK*NK+1)=C+nk*nk, C(1)=C+0
                 */
                const float *A1 = A;
                const float *A2 = notrans ? A + nk : A + (long)nk * lda;

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_UPPER, notrans ? opN : opT,
                    nk, k, alpha, A1, lda, beta, C + (long)nk * (nk + 1), nk));

                CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
                    CUBLAS_FILL_MODE_LOWER, notrans ? opN : opT,
                    nk, k, alpha, A2, lda, beta, C + (long)nk * nk, nk));

                CURFP_CHECK_CUBLAS(cublasSgemm(cb,
                    notrans ? opN : opT, notrans ? opT : opN,
                    nk, nk, k,
                    alpha, A2, lda, A1, lda,
                    beta, C + 0, nk));
            }
        }
    }

    return CURFP_STATUS_SUCCESS;
}
