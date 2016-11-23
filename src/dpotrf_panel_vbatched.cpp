/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016
       
       @author Azzam Haidar
       @author Tingxing Dong

       @generated from src/zpotrf_panel_vbatched.cpp, normal z -> d, Sun Nov 20 20:20:26 2016
*/
#include "magma_internal.h"
#include <cublas_v2.h>
#define PRECISION_d

#include "batched_kernel_param.h"
////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dpotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magma_int_t *ibvec, magma_int_t nb,  
    double** dA_array,    magma_int_t* ldda,
    double** dX_array,    magma_int_t* dX_length,
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dW0_displ, double** dW1_displ, 
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t *n_minus_ib;
    magma_imalloc( &n_minus_ib, batchCount );
    arginfo = magma_dpotf2_vbatched(
                       uplo, ibvec, nb,
                       dA_array, ldda,
                       dW1_displ, dW2_displ,
                       dW3_displ, dW4_displ,
                       info_array, gbstep,
                       batchCount, queue);

    if ((max_n-nb) > 0) {
        // n-ib
        magma_ivec_add( batchCount, 1, n, -1, ibvec, n_minus_ib, queue);
        magma_ddisplace_pointers_var_cc(dW0_displ, dA_array, ldda, nb, 0, batchCount, queue);
        magmablas_dtrsm_work_vbatched( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 1,  
                                      n_minus_ib, ibvec, 
                                      MAGMA_D_ONE, 
                                      dA_array,  ldda, 
                                      dW0_displ, ldda, 
                                      dX_array,  n_minus_ib, 
                                      dinvA_array, dinvA_length, 
                                      dW1_displ, dW2_displ, 
                                      dW3_displ, dW4_displ, 
                                      0, batchCount, 
                                      max_n-nb, nb, queue );
    }
    magma_free( n_minus_ib );
    return arginfo;
}
////////////////////////////////////////////////////////////////////////////////////////
