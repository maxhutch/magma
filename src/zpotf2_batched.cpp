/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2013
       
       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/
#include "common_magma.h"
#include "batched_kernel_param.h"

#define PRECISION_z
/////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotf2_ztrsm_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magmaDoubleComplex **dA_displ, 
    magmaDoubleComplex **dB_displ, 
    magmaDoubleComplex **dC_displ,
    magma_int_t *info_array, magma_int_t gbstep,  
    magma_int_t batchCount, magma_queue_t queue)
{


    magma_int_t j;
    magma_int_t arginfo = 0;
    if( m > MAX_NTHREADS )
    {
        printf("magma_zpotf2_ztrsm_batched m=%d > %d not supported today \n", (int) m, (int) MAX_NTHREADS);
        arginfo =-13;
        return arginfo;
    }

    // Quick return if possible
    if (n == 0) {
        return arginfo;
    }


    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else {
        for(j = 0; j < n; j++) {
            magma_zpotf2_zdotc_batched(j, dA_array, lda, j, info_array, gbstep, batchCount, queue); // including zdotc product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                magma_zlacgv_batched(j, dA_array, lda, j, batchCount, queue);
                #endif

                magma_zdisplace_pointers(dA_displ, dA_array, lda, j+1, 0, batchCount, queue);
                magma_zdisplace_pointers(dB_displ, dA_array, lda, j, 0, batchCount, queue);
                magma_zdisplace_pointers(dC_displ, dA_array, lda, j+1, j, batchCount, queue);

                // Compute elements J+1:N of column J = A(j+1:n,1:j-1) * A(j,1:j-1) (row).
                magmablas_zgemv_batched(MagmaNoTrans, m-j-1, j,
                                 alpha, dA_displ, lda,
                                        dB_displ,    lda,
                                 beta,  dC_displ, 1,
                                 batchCount, queue);// 


                #if defined(PRECISION_z) || defined(PRECISION_c)
                magma_zlacgv_batched(j, dA_array, lda, j, batchCount, queue);
                #endif
                magma_zpotf2_zdscal_batched(m-j, dA_array, 1, j+j*lda, info_array, batchCount, queue);
            }
        }
    }

    return arginfo;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magmaDoubleComplex **dA_displ, 
    magmaDoubleComplex **dW_displ,
    magmaDoubleComplex **dB_displ, 
    magmaDoubleComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{

    magma_int_t j;

    // Quick return if possible
    if (n == 0) {
        return 1;
    }

    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;


    int nb = POTF2_NB;
    int ib, rows;

    if (uplo == MagmaUpper) {
       printf("Upper side is unavailable \n");
    }
    else {
        for(j = 0; j < n; j+= nb) {
            ib   = min(nb, n-j);
            rows = m-j;
            if( (rows <= POTF2_TILE_SIZE) && (ib <= POTF2_TILE_SIZE) ){
                magma_zdisplace_pointers(dA_displ, dA_array, lda, j, j, batchCount, queue);
                magma_zpotf2_tile_batched(
                               uplo, rows, ib,
                               dA_displ, lda,
                               info_array, gbstep, batchCount, queue);
            }
            else{
                 magma_zdisplace_pointers(dA_displ, dA_array, lda, j, j, batchCount, queue); 
                 magma_zpotf2_ztrsm_batched(
                           uplo, rows, ib,
                           dA_displ, lda,
                           dW_displ, dB_displ, dC_displ, 
                           info_array, gbstep, batchCount, queue);

            }
#if 1

//#define RIGHT_LOOKING
            if( (n-j-ib) > 0){
#ifdef RIGHT_LOOKING
                magma_zdisplace_pointers(dA_displ, dA_array, lda, j+ib, j, batchCount, queue);
                magma_zdisplace_pointers(dC_displ, dA_array, lda, j+ib, j+ib, batchCount, queue);
                #if 1
                magmablas_zgemm_batched(MagmaNoTrans, MagmaConjTrans, m-j-ib, n-j-ib, ib,
                             alpha, dA_displ, lda,
                                    dA_displ, lda,
                             beta,  dC_displ, lda, batchCount, queue );
                #else
                cublasZgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_C, m-j-ib, n-j-ib, ib,
                             &alpha, (const magmaDoubleComplex**) dA_displ, lda,
                                    (const magmaDoubleComplex**) dA_displ, lda,
                             &beta,  dC_displ, lda, batchCount, queue );
                #endif
#else
                // update next subpanel
                magma_zdisplace_pointers(dA_displ, dA_array, lda, j+ib, 0, batchCount, queue);
                magma_zdisplace_pointers(dC_displ, dA_array, lda, j+ib, j+ib, batchCount, queue);
                #if 1
                magmablas_zgemm_batched(MagmaNoTrans, MagmaConjTrans, m-j-ib, min((n-j-ib),ib), j+ib,
                             alpha, dA_displ, lda,
                                    dA_displ, lda,
                             beta,  dC_displ, lda, batchCount, queue );
                #else
                cublasZgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_C, m-j-ib, min((n-j-ib),ib), j+ib,
                             &alpha, (const magmaDoubleComplex**) dA_displ, lda,
                                    (const magmaDoubleComplex**) dA_displ, lda,
                             &beta,  dC_displ, lda, batchCount );
                #endif
#endif
             } // end of if( (n-j-ib) > 0)
#endif
        }
    }

    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



