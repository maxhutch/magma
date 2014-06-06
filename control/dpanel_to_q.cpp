/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Mark Gates
       @generated d Tue Dec 17 13:18:17 2013
*/
#include "common_magma.h"

// -------------------------
// Put 0s in the upper triangular part of a panel and 1s on the diagonal.
// Stores previous values in work array, to be restored later with dq_to_panel.
extern "C"
void dpanel_to_q(char uplo, magma_int_t ib, double *A, magma_int_t lda, double *work)
{
    int i, j, k = 0;
    double *col;
    double c_zero = MAGMA_D_ZERO;
    double c_one  = MAGMA_D_ONE;
    
    if (uplo == 'U' || uplo == 'u'){
        for(i = 0; i < ib; ++i){
            col = A + i*lda;
            for(j = 0; j < i; ++j){
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
        for(i=0; i<ib; ++i){
            col = A + i*lda;
            work[k] = col[i];
            col [i] = c_one;
            ++k;
            for(j=i+1; j<ib; ++j){
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
        }
    }
}


// -------------------------
// Restores a panel, after call to dpanel_to_q.
extern "C"
void dq_to_panel(char uplo, magma_int_t ib, double *A, magma_int_t lda, double *work)
{
    int i, j, k = 0;
    double *col;
    
    if (uplo == 'U' || uplo == 'u'){
        for(i = 0; i < ib; ++i){
            col = A + i*lda;
            for(j = 0; j <= i; ++j){
                col[j] = work[k];
                ++k;
            }
        }
    }
    else {
        for(i = 0; i < ib; ++i){
            col = A + i*lda;
            for(j = i; j < ib; ++j){
                col[j] = work[k];
                ++k;
            }
        }
    }
}
