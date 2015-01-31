/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2013
       
       @author Azzam Haidar
       @author Tingxing Dong

       @generated from zpotf2_batched.cpp normal z -> s, Fri Jan 30 19:00:19 2015
*/
#include "common_magma.h"
#include "batched_kernel_param.h"

#define PRECISION_s
/////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_spotf2_strsm_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **dA_displ, 
    float **dB_displ, 
    float **dC_displ,
    magma_int_t *info_array, magma_int_t gbstep,  
    magma_int_t batchCount, magma_queue_t queue)
{


    magma_int_t j;
    magma_int_t arginfo = 0;
    if( m > MAX_NTHREADS )
    {
        printf("magma_spotf2_strsm_batched m=%d > %d not supported today \n", (int) m, (int) MAX_NTHREADS);
        arginfo =-13;
        return arginfo;
    }

    // Quick return if possible
    if (n == 0) {
        return arginfo;
    }


    float alpha = MAGMA_S_NEG_ONE;
    float beta  = MAGMA_S_ONE;

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else {
        for(j = 0; j < n; j++) {
            magma_spotf2_sdot_batched(j, dA_array, lda, j, info_array, gbstep, batchCount, queue); // including sdot product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                magma_slacgv_batched(j, dA_array, lda, j, batchCount, queue);
                #endif

                magma_sdisplace_pointers(dA_displ, dA_array, lda, j+1, 0, batchCount, queue);
                magma_sdisplace_pointers(dB_displ, dA_array, lda, j, 0, batchCount, queue);
                magma_sdisplace_pointers(dC_displ, dA_array, lda, j+1, j, batchCount, queue);

                // Compute elements J+1:N of column J = A(j+1:n,1:j-1) * A(j,1:j-1) (row).
                magmablas_sgemv_batched(MagmaNoTrans, m-j-1, j,
                                 alpha, dA_displ, lda,
                                        dB_displ,    lda,
                                 beta,  dC_displ, 1,
                                 batchCount, queue);// 


                #if defined(PRECISION_z) || defined(PRECISION_c)
                magma_slacgv_batched(j, dA_array, lda, j, batchCount, queue);
                #endif
                magma_spotf2_sscal_batched(m-j, dA_array, 1, j+j*lda, info_array, batchCount, queue);
            }
        }
    }

    return arginfo;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_spotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **dA_displ, 
    float **dW_displ,
    float **dB_displ, 
    float **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{

    magma_int_t j;

    // Quick return if possible
    if (n == 0) {
        return 1;
    }

    float alpha = MAGMA_S_NEG_ONE;
    float beta  = MAGMA_S_ONE;


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
                magma_sdisplace_pointers(dA_displ, dA_array, lda, j, j, batchCount, queue);
                magma_spotf2_tile_batched(
                               uplo, rows, ib,
                               dA_displ, lda,
                               info_array, gbstep, batchCount, queue);
            }
            else{
                 magma_sdisplace_pointers(dA_displ, dA_array, lda, j, j, batchCount, queue); 
                 magma_spotf2_strsm_batched(
                           uplo, rows, ib,
                           dA_displ, lda,
                           dW_displ, dB_displ, dC_displ, 
                           info_array, gbstep, batchCount, queue);

            }
#if 1

//#define RIGHT_LOOKING
            if( (n-j-ib) > 0){
#ifdef RIGHT_LOOKING
                magma_sdisplace_pointers(dA_displ, dA_array, lda, j+ib, j, batchCount, queue);
                magma_sdisplace_pointers(dC_displ, dA_array, lda, j+ib, j+ib, batchCount, queue);
                #if 1
                magmablas_sgemm_batched(MagmaNoTrans, MagmaConjTrans, m-j-ib, n-j-ib, ib,
                             alpha, dA_displ, lda,
                                    dA_displ, lda,
                             beta,  dC_displ, lda, batchCount, queue );
                #else
                cublasSgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_C, m-j-ib, n-j-ib, ib,
                             &alpha, (const float**) dA_displ, lda,
                                    (const float**) dA_displ, lda,
                             &beta,  dC_displ, lda, batchCount, queue );
                #endif
#else
                // update next subpanel
                magma_sdisplace_pointers(dA_displ, dA_array, lda, j+ib, 0, batchCount, queue);
                magma_sdisplace_pointers(dC_displ, dA_array, lda, j+ib, j+ib, batchCount, queue);
                #if 1
                magmablas_sgemm_batched(MagmaNoTrans, MagmaConjTrans, m-j-ib, min((n-j-ib),ib), j+ib,
                             alpha, dA_displ, lda,
                                    dA_displ, lda,
                             beta,  dC_displ, lda, batchCount, queue );
                #else
                cublasSgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_C, m-j-ib, min((n-j-ib),ib), j+ib,
                             &alpha, (const float**) dA_displ, lda,
                                    (const float**) dA_displ, lda,
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



