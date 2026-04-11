/*
 * curfpSstrttf — Copy triangular matrix from standard full to RFP format
 *
 * Direct CUDA translation of LAPACK strttf.f.
 * A is an n×n row-major matrix on device (lda >= n).
 * ARF is n*(n+1)/2 floats on device.
 *
 * 8 RFP storage variants (N parity × TRANSR N/T × UPLO L/U).
 *
 * Unified 2D kernel: one thread per triangular element of A.
 * Grid: (ceil(n/TX) × ceil(n/TY)), block: (TX × TY).
 * TX=32 keeps threads in a warp reading adjacent columns (coalesced row-major).
 *
 * Each thread:
 *   1. Computes (row, col) from its 2D thread index.
 *   2. Skips if out of bounds or not in the stored triangle.
 *   3. Computes the RFP flat index using the same block-decomposition
 *      formulas as the original per-case kernels, solved for arf_index
 *      given (row, col).
 *   4. Does one read from A and one write to arf.
 *
 * This launches ~n²/2 active threads (full triangular region), saturates
 * GPU occupancy, and produces coalesced reads from the row-major A matrix
 * (threads in a warp have adjacent col values → adjacent A[row,col] words).
 */

#include "curfp_internal.h"

#define TX 32
#define TY 8

/* =========================================================================
 * Unified 2D kernel
 *
 * Parameters:
 *   A        — n×n row-major source matrix (lower or upper triangle used)
 *   arf      — n*(n+1)/2 destination RFP array
 *   n        — matrix order
 *   lda      — leading dimension of A (>= n)
 *   nisodd   — 1 if n is odd
 *   normaltransr — 1 if TRANSR='N'
 *   lower    — 1 if UPLO='L'
 *   n1,n2,nk — block dimensions (same as LAPACK)
 *   nt       — n*(n+1)/2
 * ========================================================================= */
static __global__ void k_strttf(
    const float * __restrict__ A,
    float       * __restrict__ arf,
    int n, int lda,
    int nisodd, int normaltransr, int lower,
    int n1, int n2, int nk, int nt)
{
    int col = blockIdx.x * TX + threadIdx.x;
    int row = blockIdx.y * TY + threadIdx.y;

    if (col >= n || row >= n) return;

    /* Skip elements outside the stored triangle of A */
    if (lower) {
        if (col > row) return;   /* only lower triangle: col <= row */
    } else {
        if (row > col) return;   /* only upper triangle: row <= col */
    }

    /* Compute RFP flat index for A[row, col] in each of the 8 cases.
     *
     * The formulas below are the inverse of the original per-case kernels:
     * given (row, col), find arf_index such that
     *   arf[arf_index] = A[row, col]
     *
     * Variable naming follows LAPACK strttf.f:
     *   n1, n2 (odd case), nk=n/2 (even case).
     *   UPLO=L: n2=n/2, n1=n-n2
     *   UPLO=U: n1=n/2, n2=n-n1
     */
    long arf_idx = -1;

    if (nisodd) {
        /* ----------------------------------------------------------------
         * Odd n: A = [[T1(n1×n1), S^T], [S(n2×n1), T2(n2×n2)]]
         * where T1 occupies rows/cols 0..n1-1 and T2 rows/cols n1..n-1.
         * ---------------------------------------------------------------- */
        if (normaltransr) {
            /* TRANSR='N': arf stored as (n+1)/2 rows × ... column-major.
             * Case 1 (lower): arf is n×(n+1)/2 col-major, lda=n.
             *   T1 (lower, n1×n1): arf[j*n + i]        for j=0..n1-1, i=j..n-1  (col j of arf)
             *                      but the first loop (i=n1..n2+j) maps to the S^T/T2 contribution:
             * Original kernels:
             *   j=col (0..n2), first loop:  arf[j*n+(i-n1)] = A[n2+j, i]  i=n1..n2+j
             *   j=col (0..n2), second loop: arf[j*n+i]      = A[i, j]      i=j..n-1
             *
             * Case 2 (upper): reverse indexing from nt.
             */
            if (lower) {
                /* Case 1: TRANSR='N', UPLO='L'
                 * arf[j*n + (i-n1)] = A[n2+j, i]  for j=0..n2, i=n1..n2+j
                 * arf[j*n + i]      = A[i, j]      for j=0..n2, i=j..n-1
                 *
                 * Element A[row,col]:
                 *   Sub-block A[n2+j, i] i.e. row=n2+col_j, i=row: j=row-n2, i=col
                 *     → arf_idx = j*n + (i-n1) = (row-n2)*n + (col-n1)
                 *     valid when: col>=n1, row>=n2, row-n2==col? No, j=col(outer), i=row.
                 *
                 * Re-read: outer loop is j (column index of arf block, also col of A for 2nd loop).
                 * First loop:  j=0..n2, A[n2+j, i] for i=n1..n2+j  → row=n2+j, col_A=i
                 *   So: col=j (arf col), row_A = n2+j, col_A = i = n1..n2+j
                 *   Given A[row,col] in this sub-block:
                 *     j_arf = row - n2  (the arf column)
                 *     i_arf = col       (actual col of A, ≥n1)
                 *     arf_idx = j_arf * n + (i_arf - n1) = (row-n2)*n + (col-n1)
                 *   Condition: col >= n1 AND col <= row-n2+n2 = row AND row >= n2
                 *              i.e. col>=n1 AND col<=row AND row>=n2 AND (row-n2)<=n2
                 *              i.e. n1<=col<=row, row>=n2, row<n (always), col-n1<=row-n2
                 *
                 * Second loop: j=0..n2, A[i,j] for i=j..n-1  → row=i, col=j
                 *   Given A[row,col]: j_arf=col, i_arf=row
                 *   arf_idx = col*n + row
                 *   Condition: col<=row, col<=n2 (j goes to n2), row>=col
                 *
                 * Disambiguation: if col >= n1, it could be in either sub-block.
                 *   First loop  range: col in [n1, n2+j_arf] = [n1, n2+(row-n2)] = [n1, row]
                 *                     AND row in [n2, n-1]
                 *   Second loop range: col in [0, n2], row in [col, n-1]
                 *   Overlap: col in [n1, n2] AND row in [n2, n-1] AND col<=row
                 *     In overlap: both formulas must give the same arf slot — they don't.
                 *     Need to check: does one element appear in both loops?
                 *     First loop:  A[n2+j, i], j=col_arf, i=col_A → row_A=n2+j, col_A=i>=n1
                 *     Second loop: A[i, j], j=col_arf, i=row_A >= j → row_A>=col_arf, col_A=col_arf<=n2
                 *     These are disjoint if the A indices don't overlap.
                 *     First loop A element: (row_A, col_A) = (n2+j, i), i>=n1, i<=n2+j
                 *       → col_A >= n1
                 *     Second loop A element: (row_A, col_A) = (i, j), col_A=j<=n2
                 *       If col_A>=n1: then col_A in [n1,n2]. row_A=i>=j=col_A. row_A>=n2+col_A-j... 
                 *       Not the same element because first loop has col_A=i>=n1 and row_A=n2+j,
                 *       second loop has col_A=j<=n2, row_A=i>=j.
                 *       Can they be equal? col_A_1=i=col_A_2=j => i=j, but first loop i>=n1, j<=n2.
                 *       If i=j, row_A_1=n2+j, row_A_2=i=j. n2+j != j (n2>0). So NO overlap.
                 *
                 * So: if col >= n1 AND row >= n2 AND col >= n1 AND (row - n2) >= 0:
                 *     → first-loop element, arf_idx = (row-n2)*n + (col-n1)
                 *   else (col <= n2 AND row >= col):
                 *     → second-loop element, arf_idx = col*n + row
                 */
                if (col >= n1 && row >= n2) {
                    arf_idx = (long)(row - n2) * n + (col - n1);
                } else {
                    arf_idx = (long)col * n + row;
                }
            } else {
                /* Case 2: TRANSR='N', UPLO='U'
                 * Original kernel (reversed j): j=n-1-j_idx, ij0 = nt-(n-j)*n
                 *   First loop:  arf[ij0+i]              = A[i,j]        i=0..j
                 *   Second loop: arf[ij0+(j+1)+(l-(j-n1))] = A[j-n1,l]  l=j-n1..n1-1
                 *
                 * For A[row,col] in upper triangle (row<=col):
                 *
                 * First loop: A[i,j] → row=i, col=j, j>=n2 (since j_idx<n2 → j=n-1-j_idx>n-1-n2=n1-1)
                 *   Actually j ranges from n-1 down to n-n2=n1. So col>=n1.
                 *   ij0 = nt-(n-col)*n
                 *   arf_idx = ij0 + row = nt-(n-col)*n + row
                 *   Condition: col>=n1, row<=col
                 *   But also second loop uses same j with ij0, writing arf[ij0+(col+1)+...]
                 *   For first loop: row in [0,col], arf slot = ij0+row
                 *   For second loop: A[j-n1, l] → row_A=j-n1=col-n1, col_A=l in [j-n1,n1-1]=[col-n1,n1-1]
                 *     arf_idx = ij0+(col+1)+(l-(col-n1)) where l=col_A
                 *            = ij0+(col+1)+(col_A-col+n1)
                 *   Condition: col>=n1, col_A in [col-n1, n1-1], row_A=col-n1
                 *              col_A <= n1-1, col_A >= col-n1
                 *
                 * For A[row,col] with col<n1 (j_idx corresponds to j<n1, not reached — j goes n1..n-1)
                 *   These are in the "other" block. But wait: col can also be < n1 from second loop.
                 *   Second loop: row_A=col_arf-n1 (col_arf=j in [n1,n-1]), col_A=l in [col_arf-n1, n1-1]
                 *     col_A <= n1-1 < n1. So col_A (= actual matrix column) < n1. ✓
                 *     row_A = col_arf - n1 >= 0, row_A = col_arf-n1 <= n2-1 < n1.
                 *     row_A in [0, n2-1], col_A in [row_A, n1-1] (since col_A>=col_arf-n1=row_A)
                 *
                 * So: if col >= n1 AND row <= col (first loop):
                 *     ij0 = nt-(n-col)*n
                 *     arf_idx = ij0 + row
                 *   else (col < n1, this is the second-loop region, accessed via j=col_arf=col+n1+?):
                 *     row_A = col_arf-n1 → col_arf = row+n1   (here row_A=row, col_arf=j)
                 *     col_A = col (= l)
                 *     j = col_arf = row+n1
                 *     ij0 = nt-(n-j)*n = nt-(n-row-n1)*n = nt-(n2-row)*n
                 *     arf_idx = ij0 + (j+1) + (col - (j-n1))
                 *             = ij0 + (row+n1+1) + (col - row)
                 *             = nt-(n2-row)*n + (row+n1+1) + (col-row)
                 *             = nt-(n2-row)*n + n1+1+col
                 *   Condition for second loop: row<n2 (=n-n1), col in [row, n1-1]
                 *   (row<=col is already guaranteed by upper-triangle guard)
                 */
                if (col >= n1) {
                    long ij0 = (long)nt - (long)(n - col) * n;
                    arf_idx = ij0 + row;
                } else {
                    /* col < n1: second-loop element accessed via j=row+n1 */
                    long ij0 = (long)nt - (long)(n2 - row) * n;
                    arf_idx = ij0 + (long)(n1 + 1 + col);
                }
            }
        } else {
            /* TRANSR='T' (transposed RFP): arf stored as (n+1)/2 cols × ... row-major.
             * Case 3 (lower), Case 4 (upper).
             */
            if (lower) {
                /* Case 3: TRANSR='T', UPLO='L'
                 * arf is n1×(n+1) col-major (lda=n1).
                 * Original kernel (j=0..n-1):
                 *   ij0 = j*n1
                 *   j<n2:
                 *     arf[ij0+i] = A[j,i]   i=0..j         → row=j, col=i, i<=j
                 *     arf[ij0+(i-n1+1)] = A[i, n1+j]  i=n1+j..n-1 → row=i, col=n1+j
                 *   j>=n2:
                 *     arf[ij0+i] = A[j,i]   i=0..n1-1
                 *
                 * For A[row,col]:
                 *   Region 1 (j<n2, i<=j): row=j<n2, col=i<=row → arf_idx = row*n1 + col
                 *   Region 2 (j<n2, col=n1+j): row=i>=n1+j=n1+col_arf... 
                 *     Here col_A=n1+j → j=col-n1, must have j<n2 → col<n1+n2=n → always.
                 *     ij0=(col-n1)*n1, i=row, arf_idx=(col-n1)*n1+(row-n1+1)
                 *     Condition: col>=n1 (from col=n1+j, j>=0), row>=col (lower tri), row>=n1+j=col
                 *   Region 3 (j>=n2, i<n1): row=j>=n2, col=i<n1 → arf_idx=row*n1+col
                 *
                 *   Regions 1 and 3 have same formula arf_idx=row*n1+col:
                 *     Region 1: row<n2, col<=row
                 *     Region 3: row>=n2, col<n1
                 *   Combined: if col<n1 (and col<=row from lower-tri guard) → arf_idx=row*n1+col
                 *   Region 2: col>=n1 → arf_idx=(col-n1)*n1+(row-n1+1)
                 */
                if (col < n1) {
                    arf_idx = (long)row * n1 + col;
                } else {
                    arf_idx = (long)(col - n1) * n1 + (row - n1 + 1);
                }
            } else {
                /* Case 4: TRANSR='T', UPLO='U'  [fused]
                 * arf is n2×(n+1) col-major, effective lda depends on sub-part.
                 * Original kernel (j=0..n1):
                 *   A-part (j=0..n1): ij0=j*n2, A[j,i] i=n1..n-1 → row=j, col=i>=n1
                 *     arf_idx = j*n2 + (i-n1) = row*n2 + (col-n1)
                 *   B-part (j=0..n1-1): ij1=(n1+1)*n2 + j*(n1+1)
                 *     A[i,j] i=0..j: row=i, col=j<n1 → arf_idx=ij1+row=(n1+1)*n2+col*(n1+1)+row
                 *     A[n2+j,l] l=n2+j..n-1: row=n2+j, col=l → j=row-n2,
                 *       ij1=(n1+1)*n2+(row-n2)*(n1+1)
                 *       arf_idx=ij1+(j+1)+(l-(n2+j))=ij1+(row-n2+1)+(col-(n2+row-n2))
                 *             =ij1+(row-n2+1)+(col-row)=(n1+1)*n2+(row-n2)*(n1+1)+(row-n2+1)+(col-row)
                 *
                 * Disambiguation:
                 *   A-part: col>=n1, row<=n1 (j=row<=n1)
                 *   B-part first: col<n1, row<=col
                 *   B-part second: row=n2+j>=n2, col>=row (upper tri, col>=row), col in [row,n-1]
                 *     AND row-n2<n1 i.e. row<n1+n2=n. Always true.
                 *     col>=n2+j+(j? no)... col=l>=n2+j=row → col>=row ✓
                 *     col<n? yes. j=row-n2, l in [n2+j, n-1]=[row, n-1] → col in [row, n-1]
                 *     AND col>=n1? Not necessarily. col=l, l>=n2+j=row. col could be <n1 if row<n1.
                 *     Hmm. Let's check: B-part second A[n2+j,l] with j<n1, l in [n2+j,n-1].
                 *     col_A=l>=n2+j>=n2. col>=n2. Is n2>=n1? For odd n, uplo=U: n1=n/2, n2=n-n1=ceil(n/2)>=n1.
                 *     So col>=n2>=n1. Good.
                 *
                 * So:
                 *   if col >= n1 AND row <= n1:  A-part → arf_idx = row*n2 + (col-n1)
                 *   elif col < n1 AND row <= col: B-part first → arf_idx = (n1+1)*n2 + col*(n1+1) + row
                 *   else (col>=n1, row>=n2, col>=row): B-part second
                 *     j=row-n2
                 *     ij1=(n1+1)*n2+(row-n2)*(n1+1)
                 *     arf_idx=ij1+(row-n2+1)+(col-row)
                 */
                if (col >= n1 && row <= n1) {
                    arf_idx = (long)row * n2 + (col - n1);
                } else if (col < n1) {
                    /* B-part first: col<n1, row<=col (upper tri) */
                    arf_idx = (long)(n1 + 1) * n2 + (long)col * (n1 + 1) + row;
                } else {
                    /* B-part second: col>=n1, row>=n2 */
                    long ij1 = (long)(n1 + 1) * n2 + (long)(row - n2) * (n1 + 1);
                    arf_idx = ij1 + (row - n2 + 1) + (col - row);
                }
            }
        }
    } else {
        /* Even n */
        if (normaltransr) {
            if (lower) {
                /* Case 5: TRANSR='N', UPLO='L'
                 * arf is (n+1)×nk col-major, lda=n+1.
                 * Original kernel (j=0..nk-1):
                 *   ij0 = j*(n+1)
                 *   First loop:  arf[ij0+(i-nk)] = A[nk+j, i]  i=nk..nk+j  → row=nk+j, col=i>=nk
                 *   Second loop: arf[ij0+i+1]    = A[i, j]      i=j..n-1    → row=i, col=j<nk
                 *
                 * For A[row,col]:
                 *   First loop: col>=nk, row=nk+j → j=row-nk, col=i in [nk,nk+j]=[nk,row]
                 *     → j_arf=row-nk, ij0=(row-nk)*(n+1), arf_idx=ij0+(col-nk)=(row-nk)*(n+1)+(col-nk)
                 *     Condition: col>=nk, row>=nk, col<=row
                 *   Second loop: col=j<nk, row=i>=j=col
                 *     ij0=col*(n+1), arf_idx=col*(n+1)+row+1
                 *     Condition: col<nk, row>=col
                 */
                if (col >= nk) {
                    arf_idx = (long)(row - nk) * (n + 1) + (col - nk);
                } else {
                    arf_idx = (long)col * (n + 1) + row + 1;
                }
            } else {
                /* Case 6: TRANSR='N', UPLO='U'
                 * arf is (n+1)×nk col-major, lda=n+1.
                 * Original kernel (j_idx=0..nk-1, j=nk+j_idx):
                 *   ij0 = nt-(n+1)*(n-j)
                 *   First loop:  arf[ij0+i]        = A[i,j]    i=0..j       → row=i, col=j>=nk
                 *   Second loop: arf[ij0+nk+1+l]   = A[j-nk,l] l=j-nk..nk-1 → row=j-nk, col=l<nk
                 *     (since l<=nk-1<nk)
                 *
                 * For A[row,col]:
                 *   First loop: col>=nk, row<=col, j=col
                 *     ij0=nt-(n+1)*(n-col), arf_idx=ij0+row=nt-(n+1)*(n-col)+row
                 *   Second loop: col<nk, row=j-nk → j=row+nk, col=l in [j-nk,nk-1]=[row,nk-1]
                 *     ij0=nt-(n+1)*(n-j)=nt-(n+1)*(n-row-nk)=nt-(n+1)*(nk-row)
                 *     arf_idx=ij0+nk+1+col=nt-(n+1)*(nk-row)+nk+1+col
                 *     Condition: col<nk, row<=col (upper tri → row<=col ✓ since col=l>=row)
                 */
                if (col >= nk) {
                    arf_idx = (long)nt - (long)(n + 1) * (n - col) + row;
                } else {
                    arf_idx = (long)nt - (long)(n + 1) * (nk - row) + (nk + 1 + col);
                }
            }
        } else {
            /* TRANSR='T' */
            if (lower) {
                /* Case 7: TRANSR='T', UPLO='L'  [fused]
                 * arf is nk×(n+1) col-major, first element arf[0] is special.
                 * Original kernel (j=0..n-1):
                 *   col part (j<nk): arf[j] = A[nk+j, nk]  → row=nk+j, col=nk
                 *     → arf_idx = row-nk (= j)
                 *   main part, ij0=nk*(j+1):
                 *     j<nk-1:
                 *       arf[ij0+i]    = A[j,i]         i=0..j    → row=j, col=i, col<=row, row<nk-1
                 *       arf[ij0+(i-nk)] = A[i,nk+1+j]  i=nk+1+j..n-1 → row=i, col=nk+1+j
                 *     j>=nk-1 (i.e. j=nk-1..n-1):
                 *       arf[ij0+i]    = A[j,i]         i=0..nk-1 → row=j>=nk-1, col=i<nk
                 *
                 * The col-part writes arf[0..nk-1] for A[nk..n-1, nk] (col=nk exactly).
                 * The main part uses ij0=nk*(j+1)>=nk, so arf[nk..] — disjoint.
                 *
                 * For A[row,col]:
                 *   col-part: col=nk, row>=nk → arf_idx=row-nk
                 *   main j<nk-1, first loop: row=j<nk-1, col<=row<nk-1 (col<nk)
                 *     ij0=nk*(row+1), arf_idx=ij0+col=nk*(row+1)+col
                 *   main j<nk-1, second loop: col=nk+1+j → j=col-nk-1<nk-1 → col<2nk=n, col>=nk+1
                 *     row>=col, ij0=nk*(j+1)=nk*(col-nk)
                 *     arf_idx=ij0+(row-nk)=nk*(col-nk)+(row-nk)
                 *   main j>=nk-1, row=j>=nk-1, col=i<nk:
                 *     ij0=nk*(row+1), arf_idx=ij0+col=nk*(row+1)+col
                 *     (same formula as first-loop case!)
                 *
                 * Combined:
                 *   col=nk exactly AND row>=nk: arf_idx=row-nk   (col-part)
                 *   col<nk (and col<=row from lower tri):         arf_idx=nk*(row+1)+col
                 *   col>=nk+1 (col>nk) AND row>=col:             arf_idx=nk*(col-nk)+(row-nk)
                 *
                 * Note: col=nk AND row<nk: not in lower triangle (row<col=nk), skip — handled by guard.
                 * col=nk AND row=nk: row=col=nk, lower tri → col<=row ✓.
                 *   This is the diagonal element A[nk,nk].
                 *   col-part: j=row-nk=0, arf[0]=A[nk,nk]? No: col-part writes A[nk+j,nk]=A[nk,nk] for j=0. ✓
                 *   So arf_idx=row-nk=0. ✓
                 */
                if (col == nk && row >= nk) {
                    arf_idx = row - nk;
                } else if (col < nk) {
                    arf_idx = (long)nk * (row + 1) + col;
                } else {
                    /* col > nk */
                    arf_idx = (long)nk * (col - nk) + (row - nk);
                }
            } else {
                /* Case 8: TRANSR='T', UPLO='U'  [fused]
                 * arf is (nk+1)×nk col-major-ish.
                 * Original kernel (j=0..nk):
                 *   A-part: ij0=j*nk, A[j,i] i=nk..n-1 → row=j<=nk, col=i>=nk
                 *     arf_idx=j*nk+(i-nk)=row*nk+(col-nk)
                 *   B-part (j<nk): ij1=nk*(nk+1)+j*nk
                 *     A[i,j] i=0..j: row=i<=j=col, col=j<nk → arf_idx=ij1+row=nk*(nk+1)+col*nk+row
                 *     A[nk+1+j,l] l=nk+1+j..n-1: row=nk+1+j, col=l>=row
                 *       j=row-nk-1, ij1=nk*(nk+1)+(row-nk-1)*nk
                 *       arf_idx=ij1+(j+1)+(l-(nk+1+j))=ij1+(row-nk)+(col-row)
                 *             =nk*(nk+1)+(row-nk-1)*nk+(row-nk)+(col-row)
                 *
                 * Disambiguation:
                 *   A-part: col>=nk, row<=nk (j=row<=nk)
                 *   B-part first: col<nk, row<=col<nk
                 *   B-part second: row=nk+1+j>=nk+1, col=l>=row>nk → col>=nk
                 *     AND row>nk AND j=row-nk-1<nk → row<2nk+1=n+1 → always
                 *
                 * So:
                 *   col>=nk AND row<=nk: A-part
                 *   col<nk (row<=col<nk from upper tri + col<nk): B-part first
                 *   col>=nk AND row>=nk+1: B-part second
                 *   (col>=nk AND row=nk is covered by A-part since row<=nk ✓)
                 */
                if (col >= nk && row <= nk) {
                    arf_idx = (long)row * nk + (col - nk);
                } else if (col < nk) {
                    arf_idx = (long)nk * (nk + 1) + (long)col * nk + row;
                } else {
                    /* col>=nk, row>=nk+1: B-part second */
                    long ij1 = (long)nk * (nk + 1) + (long)(row - nk - 1) * nk;
                    arf_idx = ij1 + (row - nk) + (col - row);
                }
            }
        }
    }

    if (arf_idx >= 0)
        arf[arf_idx] = A[(long)row * lda + col];
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

    int n1 = 0, n2 = 0, nk = 0;
    if (lower) { n2 = n / 2; n1 = n - n2; }
    else       { n1 = n / 2; n2 = n - n1; }
    if (!nisodd) nk = n / 2;
    int nt = n * (n + 1) / 2;

    dim3 block(TX, TY);
    dim3 grid((n + TX - 1) / TX, (n + TY - 1) / TY);

    k_strttf<<<grid, block, 0, stream>>>(
        A, arf, n, lda, nisodd, normaltransr, lower, n1, n2, nk, nt);

    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) return CURFP_STATUS_EXECUTION_FAILED;

    return CURFP_STATUS_SUCCESS;
}
