/*
 * curfpSpftri — Compute inverse of an SPD matrix from its RFP Cholesky factor.
 *
 * Implements LAPACK SPFTRI: given the RFP Cholesky factor produced by spftrf,
 * overwrites the RFP array with A^{-1} (also stored in RFP format, same
 * triangle/transr convention).
 *
 * Algorithm — direct sub-block inversion (no unpacking to full n×n):
 *
 * The Cholesky factor decomposes into three sub-blocks in the same 2×2 block
 * structure as spftrf / spftrs:
 *
 *   block11: dim11×dim11 triangular factor
 *   block22: dim22×dim22 triangular factor (Schur complement)
 *   S-block: off-diagonal factor
 *
 * Using block matrix inversion (where block22 = Cholesky of Schur complement
 * of A11):
 *
 *   A^{-1}[2,2] = SPOTRI(block22)                         [dim22×dim22]
 *   A^{-1}[1,1] = SPOTRI(block11) + H^T * H               [dim11×dim11]
 *   A^{-1}[2,1] = -P22 * G  (side=RIGHT)                  [dim22×dim11]
 *             or  -G * P22  (side=LEFT, G = A11^{-1}*A12)
 *
 * where P22 = A^{-1}[2,2], G is computed from S_block by a single STRSM
 * using the same side/fill as spftrf's STRSM but with the OPPOSITE transposition:
 *
 *   G = STRSM(trsm_side, trsm_fill, op_opp(trsm_op), block11, copy_of_S_block)
 *
 * For side=RIGHT: G = A21 * A11^{-1}  (dim22 × dim11)
 * For side=LEFT:  G = A11^{-1} * A12  (dim11 × dim22)
 *
 * H = L22^{-1} * G or G * L22^{-T} (see below) is computed BEFORE SPOTRI(block22)
 * destroys L22.  Then SSYRK(fill11) adds H^T*H (or H*H^T) to block11, which
 * only writes to the fill11 triangle and avoids corrupting block22 elements
 * that physically share the same RFP array slots as block11's padding.
 *
 * H computation (so that G^T*P22*G = H^T*H or G*P22*G^T = H*H^T):
 *   side=RIGHT, fill22=UPPER: H = U22^{-T}*G → STRSM(L,U,T, block22, copy_G)
 *   side=RIGHT, fill22=LOWER: H = L22^{-1}*G → STRSM(L,L,N, block22, copy_G)
 *   side=LEFT,  fill22=UPPER: H = G*U22^{-1} → STRSM(R,U,N, block22, copy_G)
 *   side=LEFT,  fill22=LOWER: H = G*L22^{-T} → STRSM(R,L,T, block22, copy_G)
 * Unified: h_side = opp(trsm_side);
 *          op_h   = ((fill22==UPPER) XOR (side==LEFT)) ? TRANS : NOTRANS
 *
 * Memory: 2 × trsm_m × trsm_n floats (g_buf + h_buf) + SPOTRI workspace.
 *
 * For n=1: in-place inversion using a single-thread CUDA kernel (no host copy).
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Per-case parameters — identical layout to spftrf_params_t.
 * Duplicated here so spftri has no link-time dependency on spftrf.
 * ------------------------------------------------------------------------- */
typedef struct {
    cublasFillMode_t  fill11; int dim11; long off11; int lda11;

    cublasSideMode_t  trsm_side;
    cublasFillMode_t  trsm_fill;
    cublasOperation_t trsm_op;
    long              trsm_a_off;
    long              trsm_b_off;
    int               trsm_m, trsm_n;
    int               trsm_lda, trsm_ldb;

    cublasFillMode_t  fill22; int dim22; long off22; int lda22;
} spftri_params_t;

static void get_spftri_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               spftri_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = 0;    p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = 0; p->trsm_b_off = n1; p->trsm_lda = n; p->trsm_ldb = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n;  p->lda22 = n;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = n1; p->off11 = n2; p->lda11 = n;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = n2; p->trsm_b_off = 0; p->trsm_lda = n; p->trsm_ldb = n;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = n2; p->off22 = n1; p->lda22 = n;
            }
        } else {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = 0; p->lda11 = n1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = n1; p->trsm_n = n2;
                p->trsm_a_off = 0; p->trsm_b_off = (long)n1*n1; p->trsm_lda = n1; p->trsm_ldb = n1;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = 1; p->lda22 = n1;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = n1; p->off11 = (long)n2*n2;  p->lda11 = n2;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = n2; p->trsm_n = n1;
                p->trsm_a_off = (long)n2*n2; p->trsm_b_off = 0; p->trsm_lda = n2; p->trsm_ldb = n2;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = n2; p->off22 = (long)n1*n2; p->lda22 = n2;
            }
        }
    } else {
        if (normaltransr) {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = 1;    p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = 1; p->trsm_b_off = nk+1; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = 0;    p->lda22 = n+1;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_LOWER; p->dim11 = nk; p->off11 = nk+1; p->lda11 = n+1;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_LOWER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk+1; p->trsm_b_off = 0; p->trsm_lda = n+1; p->trsm_ldb = n+1;
                p->fill22 = CUBLAS_FILL_MODE_UPPER; p->dim22 = nk; p->off22 = nk;   p->lda22 = n+1;
            }
        } else {
            if (lower) {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = nk;           p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_LEFT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_T; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = nk; p->trsm_b_off = (long)nk*(nk+1); p->trsm_lda = nk; p->trsm_ldb = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = 0;            p->lda22 = nk;
            } else {
                p->fill11 = CUBLAS_FILL_MODE_UPPER; p->dim11 = nk; p->off11 = (long)nk*(nk+1); p->lda11 = nk;
                p->trsm_side = CUBLAS_SIDE_RIGHT; p->trsm_fill = CUBLAS_FILL_MODE_UPPER;
                p->trsm_op = CUBLAS_OP_N; p->trsm_m = nk; p->trsm_n = nk;
                p->trsm_a_off = (long)nk*(nk+1); p->trsm_b_off = 0; p->trsm_lda = nk; p->trsm_ldb = nk;
                p->fill22 = CUBLAS_FILL_MODE_LOWER; p->dim22 = nk; p->off22 = (long)nk*nk;     p->lda22 = nk;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Kernels
 * ------------------------------------------------------------------------- */

/* n=1 special case: arf[0] = L (Cholesky factor).  A^{-1} = 1/L^2. */
__global__ static void k_inv_sq(float *x) { x[0] = 1.0f / (x[0] * x[0]); }

/* Copy m×n matrix src (leading dim src_ld) → dst (leading dim dst_ld). */
__global__ static void k_copy_mat(const float *src, int src_ld,
                                   float *dst, int dst_ld,
                                   int m, int n)
{
    int r = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r < m && c < n) dst[(long)c * dst_ld + r] = src[(long)c * src_ld + r];
}

/* =========================================================================
 * Public entry point
 * ========================================================================= */
extern "C"
curfpStatus_t curfpSpftri(curfpHandle_t    handle,
                           curfpOperation_t transr,
                           curfpFillMode_t  uplo,
                           int              n,
                           float           *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0) return CURFP_STATUS_SUCCESS;

    cublasHandle_t     cb = handle->cublas;
    cusolverDnHandle_t cs = handle->cusolver;

    /* n=1: RFP has one element L (Cholesky factor).  A^{-1} = 1/L^2. */
    if (n == 1) {
        cudaStream_t stream;
        if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS)
            return CURFP_STATUS_EXECUTION_FAILED;
        k_inv_sq<<<1, 1, 0, stream>>>(arf);
        return (cudaGetLastError() == cudaSuccess) ? CURFP_STATUS_SUCCESS
                                                   : CURFP_STATUS_EXECUTION_FAILED;
    }

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
    spftri_params_t p;
    get_spftri_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    int m     = p.trsm_m;   /* rows of off-diagonal S block */
    int n_blk = p.trsm_n;   /* cols of off-diagonal S block */

    /* --- Query workspace for both SPOTRI calls ----------------------------- */
    int lwork11 = 0, lwork22 = 0, lwork;
    {
        cusolverStatus_t s;
        s = cusolverDnSpotri_bufferSize(cs, p.fill11, p.dim11,
                                        arf + p.off11, p.lda11, &lwork11);
        if (s != CUSOLVER_STATUS_SUCCESS) return from_cusolver_status(s);
        s = cusolverDnSpotri_bufferSize(cs, p.fill22, p.dim22,
                                        arf + p.off22, p.lda22, &lwork22);
        if (s != CUSOLVER_STATUS_SUCCESS) return from_cusolver_status(s);
    }
    lwork = (lwork11 > lwork22) ? lwork11 : lwork22;
    if (lwork < 1) lwork = 1;

    float *work    = NULL;
    float *g_buf   = NULL;  /* G = A21*A11^{-1} or A11^{-1}*A12: m × n_blk */
    float *h_buf   = NULL;  /* H = L22^{-?}*G:  same shape m × n_blk        */
    int   *devInfo = NULL;

    curfpStatus_t status = CURFP_STATUS_SUCCESS;
    int h_info = 0;
    const float one  =  1.0f;
    const float mone = -1.0f;
    const float zero =  0.0f;

/* Local error macros that jump to cleanup */
#define CHK_CS(expr) \
    do { cusolverStatus_t _s = (expr); \
         if (_s != CUSOLVER_STATUS_SUCCESS) { \
             status = from_cusolver_status(_s); goto cleanup; } \
    } while (0)
#define CHK_CB(expr) \
    do { cublasStatus_t _s = (expr); \
         if (_s != CUBLAS_STATUS_SUCCESS) { \
             status = from_cublas_status(_s); goto cleanup; } \
    } while (0)
#define CHK_CU(expr) \
    do { cudaError_t _e = (expr); \
         if (_e != cudaSuccess) { \
             status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; } \
    } while (0)

    CHK_CU(cudaMalloc((void **)&work,    (size_t)lwork     * sizeof(float)));
    CHK_CU(cudaMalloc((void **)&g_buf,   (size_t)m * n_blk * sizeof(float)));
    CHK_CU(cudaMalloc((void **)&h_buf,   (size_t)m * n_blk * sizeof(float)));
    CHK_CU(cudaMalloc((void **)&devInfo, sizeof(int)));

    /* ---- Step 1: Copy S_block → g_buf
     *
     * g_buf starts as a copy of the off-diagonal S_block element, which will be
     * transformed into G by the STRSM in Step 2.  We copy first because
     * S_block lives inside the arf array and will be overwritten in Step 5.
     * -------------------------------------------------------------------- */
    {
        cudaStream_t stream;
        if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS) {
            status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup;
        }
        dim3 blk(16, 16);
        dim3 grd((n_blk + 15)/16, (m + 15)/16);
        k_copy_mat<<<grd, blk, 0, stream>>>(
            arf + p.trsm_b_off, p.trsm_ldb,
            g_buf, m,
            m, n_blk);
        CHK_CU(cudaGetLastError());
    }

    /* ---- Step 2: Compute G in g_buf via one STRSM
     *
     * After spftrf, S_block = STRSM(trsm_side, trsm_fill, trsm_op, block11, A21).
     * To recover G = A21 * A11^{-1} (side=RIGHT) or A11^{-1} * A12 (side=LEFT),
     * apply the SAME side/fill but the OPPOSITE op:
     *
     *   side=RIGHT, trsm_op=T → apply op=N: g_buf ← g_buf * L11^{-1}
     *   side=RIGHT, trsm_op=N → apply op=T: g_buf ← g_buf * L11^{-T}
     *   side=LEFT,  trsm_op=T → apply op=N: g_buf ← U11^{-1} * g_buf
     *   side=LEFT,  trsm_op=N → apply op=T: g_buf ← L11^{-T} * g_buf
     * -------------------------------------------------------------------- */
    {
        cublasOperation_t op_opp = (p.trsm_op == CUBLAS_OP_T) ? CUBLAS_OP_N : CUBLAS_OP_T;
        CHK_CB(cublasStrsm(cb,
            p.trsm_side, p.trsm_fill, op_opp, CUBLAS_DIAG_NON_UNIT,
            m, n_blk, &one,
            arf + p.trsm_a_off, p.trsm_lda,
            g_buf, m));
    }

    /* ---- Step 3: Compute H in h_buf via one STRSM on block22
     *
     * H is chosen so that G^T*P22*G = H^T*H (side=RIGHT) or G*P22*G^T = H*H^T
     * (side=LEFT), where P22 = (L22*L22^T)^{-1}.  This factorization means:
     *
     *   side=RIGHT: P22 = L22^{-T}*L22^{-1}
     *               G^T*P22*G = (L22^{-1}*G)^T*(L22^{-1}*G) = H^T*H
     *     fill22=UPPER (U): H = U^{-T}*G → STRSM(L,U,T, block22, copy_G→h_buf)
     *     fill22=LOWER (L): H = L^{-1}*G → STRSM(L,L,N, block22, copy_G→h_buf)
     *
     *   side=LEFT: P22 = L22^{-T}*L22^{-1}
     *              G*P22*G^T = (G*L22^{-1})*(G*L22^{-1})^T = H*H^T
     *     fill22=UPPER (U): H = G*U^{-1} → STRSM(R,U,N, block22, copy_G→h_buf)
     *     fill22=LOWER (L): H = G*L^{-T} → STRSM(R,L,T, block22, copy_G→h_buf)
     *
     * Unified:
     *   h_side = opp(trsm_side)
     *   op_h   = ((fill22==UPPER) XOR (side==LEFT)) ? TRANS : NOTRANS
     *
     * We copy g_buf → h_buf first (STRSM overwrites in-place).
     * This must be done BEFORE SPOTRI(block22) destroys L22.
     * -------------------------------------------------------------------- */
    {
        cudaStream_t stream;
        if (cublasGetStream(cb, &stream) != CUBLAS_STATUS_SUCCESS) {
            status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup;
        }
        dim3 blk(16, 16);
        dim3 grd((n_blk + 15)/16, (m + 15)/16);
        k_copy_mat<<<grd, blk, 0, stream>>>(g_buf, m, h_buf, m, m, n_blk);
        CHK_CU(cudaGetLastError());

        int fill22_upper = (p.fill22 == CUBLAS_FILL_MODE_UPPER);
        int side_left    = (p.trsm_side == CUBLAS_SIDE_LEFT);
        cublasOperation_t op_h   = ((fill22_upper ^ side_left)) ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasSideMode_t  h_side = side_left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;

        CHK_CB(cublasStrsm(cb,
            h_side, p.fill22, op_h, CUBLAS_DIAG_NON_UNIT,
            m, n_blk, &one,
            arf + p.off22, p.lda22,
            h_buf, m));
    }

    /* ---- Step 4: SPOTRI on block22 → P22 = A^{-1}[2,2] ---------------- */
    CHK_CU(cudaMemset(devInfo, 0, sizeof(int)));
    CHK_CS(cusolverDnSpotri(cs, p.fill22, p.dim22,
                             arf + p.off22, p.lda22, work, lwork, devInfo));
    CHK_CU(cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) { status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; }

    /* ---- Step 5: SPOTRI on block11 → A11^{-1} (intermediate) ---------- */
    CHK_CU(cudaMemset(devInfo, 0, sizeof(int)));
    CHK_CS(cusolverDnSpotri(cs, p.fill11, p.dim11,
                             arf + p.off11, p.lda11, work, lwork, devInfo));
    CHK_CU(cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) { status = CURFP_STATUS_EXECUTION_FAILED; goto cleanup; }

    /* ---- Step 6: Correct block11 using SSYRK(fill11)
     *
     * block11[fill11] += H^T*H (side=RIGHT, H is m × n_blk = dim22 × dim11)
     * block11[fill11] += H*H^T (side=LEFT,  H is m × n_blk = dim11 × dim22)
     *
     * SSYRK only writes to the fill11 triangle, so it cannot corrupt block22
     * elements that physically share RFP array slots with block11's padding.
     * -------------------------------------------------------------------- */
    if (p.trsm_side == CUBLAS_SIDE_RIGHT) {
        /* H^T * H: n=dim11=n_blk, k=dim22=m */
        CHK_CB(cublasSsyrk(cb,
            p.fill11, CUBLAS_OP_T,
            n_blk, m,
            &one,  h_buf, m,
            &one,  arf + p.off11, p.lda11));
    } else {
        /* H * H^T: n=dim11=m, k=dim22=n_blk */
        CHK_CB(cublasSsyrk(cb,
            p.fill11, CUBLAS_OP_N,
            m, n_blk,
            &one,  h_buf, m,
            &one,  arf + p.off11, p.lda11));
    }

    /* ---- Step 7: Compute off-diagonal A^{-1}[2,1] (or A^{-1}[1,2])
     *
     * For side=RIGHT (G = A21 * A11^{-1}, dim22×dim11):
     *   S_block ← -P22 * G   (SSYMM LEFT, P22 symmetric dim22×dim22)
     *
     * For side=LEFT (G = A11^{-1} * A12, dim11×dim22):
     *   S_block ← -G * P22   (SSYMM RIGHT, P22 symmetric dim22×dim22)
     *
     * In both cases: S_block = A^{-1}[off-diagonal] in the RFP convention.
     * -------------------------------------------------------------------- */
    if (p.trsm_side == CUBLAS_SIDE_RIGHT) {
        CHK_CB(cublasSsymm(cb,
            CUBLAS_SIDE_LEFT, p.fill22,
            m, n_blk, &mone,
            arf + p.off22, p.lda22,          /* P22 (m × m = dim22 × dim22) */
            g_buf, m,                          /* G (m × n_blk) */
            &zero, arf + p.trsm_b_off, p.trsm_ldb));
    } else {
        CHK_CB(cublasSsymm(cb,
            CUBLAS_SIDE_RIGHT, p.fill22,
            m, n_blk, &mone,
            arf + p.off22, p.lda22,          /* P22 (n_blk × n_blk = dim22 × dim22) */
            g_buf, m,                          /* G (m × n_blk) */
            &zero, arf + p.trsm_b_off, p.trsm_ldb));
    }

#undef CHK_CS
#undef CHK_CB
#undef CHK_CU

cleanup:
    cudaFree(work);
    cudaFree(g_buf);
    cudaFree(h_buf);
    cudaFree(devInfo);
    return status;
}
