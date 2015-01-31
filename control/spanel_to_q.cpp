/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @generated from zpanel_to_q.cpp normal z -> s, Fri Jan 30 19:00:20 2015
*/
#include "common_magma.h"

// -------------------------
// Put 0s in the upper triangular part of a panel and 1s on the diagonal.
// Stores previous values in work array, to be restored later with sq_to_panel.
extern "C"
void spanel_to_q(magma_uplo_t uplo, magma_int_t ib, float *A, magma_int_t lda, float *work)
{
    magma_int_t i, j, k = 0;
    float *col;
    float c_zero = MAGMA_S_ZERO;
    float c_one  = MAGMA_S_ONE;
    
    if (uplo == MagmaUpper) {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = 0; j < i; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
            
            work[k] = col[i];
            col [j] = c_one;
            ++k;
        }
    }
    else {
        for(i=0; i<ib; ++i) {
            col = A + i*lda;
            work[k] = col[i];
            col [i] = c_one;
            ++k;
            for(j=i+1; j<ib; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
        }
    }
}


// -------------------------
// Restores a panel, after call to spanel_to_q.
extern "C"
void sq_to_panel(magma_uplo_t uplo, magma_int_t ib, float *A, magma_int_t lda, float *work)
{
    magma_int_t i, j, k = 0;
    float *col;
    
    if (uplo == MagmaUpper) {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = 0; j <= i; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
    else {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = i; j < ib; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
}
