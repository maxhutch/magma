/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotf2_batched.cpp normal z -> c, Mon May  2 23:30:27 2016
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define COMPLEX

/////////////////////////////////////////////////////////////////
/**
    \n
    This is an internal routine.
    ********************************************************************/
extern "C" magma_int_t
magma_cpotf2_ctrsm_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dA_displ, 
    magmaFloatComplex **dB_displ, 
    magmaFloatComplex **dC_displ,
    magma_int_t *info_array, magma_int_t gbstep,  
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t j;
    magma_int_t arginfo = 0;
    if ( m > MAX_NTHREADS )
    {
        printf("magma_cpotf2_ctrsm_batched m=%d > %d not supported today\n", (int) m, (int) MAX_NTHREADS);
        arginfo = -13;
        return arginfo;
    }

    // Quick return if possible
    if (n == 0) {
        return arginfo;
    }

    magmaFloatComplex alpha = MAGMA_C_NEG_ONE;
    magmaFloatComplex beta  = MAGMA_C_ONE;

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else {
        for (j = 0; j < n; j++) {
            magma_cpotf2_cdotc_batched(j, dA_array, lda, j, info_array, gbstep, batchCount, queue); // including cdotc product and update a(j,j)
            if (j < n) {
                #ifdef COMPLEX
                magma_clacgv_batched(j, dA_array, lda, j, batchCount, queue);
                #endif

                magma_cdisplace_pointers(dA_displ, dA_array, lda, j+1, 0, batchCount, queue);
                magma_cdisplace_pointers(dB_displ, dA_array, lda, j, 0, batchCount, queue);
                magma_cdisplace_pointers(dC_displ, dA_array, lda, j+1, j, batchCount, queue);

                // Compute elements J+1:N of column J = A(j+1:n,1:j-1) * A(j,1:j-1) (row).
                magmablas_cgemv_batched( MagmaNoTrans, m-j-1, j,
                                 alpha, dA_displ, lda,
                                        dB_displ,    lda,
                                 beta,  dC_displ, 1,
                                 batchCount, queue );

                #ifdef COMPLEX
                magma_clacgv_batched(j, dA_array, lda, j, batchCount, queue);
                #endif
                magma_cpotf2_csscal_batched(m-j, dA_array, 1, j+j*lda, info_array, batchCount, queue);
            }
        }
    }

    return arginfo;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    \n
    This is an internal routine.
    ********************************************************************/
extern "C" magma_int_t
magma_cpotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dA_displ, 
    magmaFloatComplex **dW_displ,
    magmaFloatComplex **dB_displ, 
    magmaFloatComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo=0;

    // Quick return if possible
    if (n == 0) {
        return 1;
    }

    magmaFloatComplex alpha = MAGMA_C_NEG_ONE;
    magmaFloatComplex beta  = MAGMA_C_ONE;


    magma_int_t nb = POTF2_NB;
    magma_int_t j, ib, rows;
    magma_int_t crossover = magma_get_cpotrf_batched_crossover();

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else {
        if ( n <= crossover )
        {
            arginfo = magma_cpotrf_lpout_batched(uplo, n, dA_array, lda, gbstep, info_array, batchCount, queue);
        } else {
            for (j = 0; j < n; j += nb) {
                ib   = min(nb, n-j);
                rows = m-j;
                if ( (rows <= POTF2_TILE_SIZE) && (ib <= POTF2_TILE_SIZE) ) {
                    magma_cdisplace_pointers(dA_displ, dA_array, lda, j, j, batchCount, queue);
                    arginfo = magma_cpotf2_tile_batched(
                                   uplo, rows, ib,
                                   dA_displ, lda,
                                   info_array, gbstep, batchCount, queue);
                }
                else {
                    magma_cdisplace_pointers(dA_displ, dA_array, lda, j, j, batchCount, queue); 
                    magma_cpotf2_ctrsm_batched(
                              uplo, rows, ib,
                              dA_displ, lda,
                              dW_displ, dB_displ, dC_displ, 
                              info_array, gbstep, batchCount, queue);
                }
                #if 1
                //#define RIGHT_LOOKING
                if ( (n-j-ib) > 0) {
                    #ifdef RIGHT_LOOKING
                    magma_cdisplace_pointers(dA_displ, dA_array, lda, j+ib, j, batchCount, queue);
                    magma_cdisplace_pointers(dC_displ, dA_array, lda, j+ib, j+ib, batchCount, queue);
                    magma_cgemm_batched( MagmaNoTrans, MagmaConjTrans,
                                 m-j-ib, n-j-ib, ib,
                                 alpha, dA_displ, lda,
                                        dA_displ, lda,
                                 beta,  dC_displ, lda, batchCount, queue );
                #else
                    // update next subpanel
                    magma_cdisplace_pointers(dA_displ, dA_array, lda, j+ib, 0, batchCount, queue);
                    magma_cdisplace_pointers(dC_displ, dA_array, lda, j+ib, j+ib, batchCount, queue);
                    magma_cgemm_batched( MagmaNoTrans, MagmaConjTrans,
                                 m-j-ib, min((n-j-ib),ib), j+ib,
                                 alpha, dA_displ, lda,
                                        dA_displ, lda,
                                 beta,  dC_displ, lda, batchCount, queue );
                #endif
                } // end of if ( (n-j-ib) > 0)
                #endif
            }
        }
    }

    return arginfo;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
