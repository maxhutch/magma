/*
   -- MAGMA (version 2.0.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date May 2016

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from src/zgeqrf_panel_batched.cpp normal z -> d, Mon May  2 23:30:28 2016
   */
#include "magma_internal.h"



////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dgeqrf_panel_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,    
        double** dA_array,    magma_int_t ldda,
        double** tau_array, 
        double** dT_array, magma_int_t ldt, 
        double** dR_array, magma_int_t ldr,
        double** dW0_displ, 
        double** dW1_displ,
        double   *dwork,  
        double** dW2_displ, 
        double** dW3_displ,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t j, jb;
    magma_int_t ldw = nb; 
    magma_int_t minmn = min(m,n); 

    for( j=0; j < minmn; j += nb)
    {
        jb = min(nb, minmn-j);

        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
        magma_ddisplace_pointers(dW2_displ, tau_array, 1, j, 0, batchCount, queue);
        magma_ddisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue); // 

        //sub-panel factorization 
        magma_dgeqr2_batched(
                m-j, jb,
                dW0_displ, ldda,      
                dW2_displ, 
                info_array, 
                batchCount,
                queue);

        //copy th whole rectangular n,jb from of dA to dR (it's lower portion (which is V's) will be set to zero if needed at the end)
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, 0, j, batchCount, queue); 
        magma_ddisplace_pointers(dW3_displ, dR_array, ldr, 0, j, batchCount, queue); 
        magmablas_dlacpy_batched( MagmaFull, minmn, jb, dW0_displ, ldda, dW3_displ, ldr, batchCount, queue );

        //set the upper jbxjb portion of V dA(j,j) to 1/0s (note that the rectangular on the top of this triangular of V still non zero but has been copied to dR).
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
        magmablas_dlaset_batched( MagmaUpper, jb, jb, MAGMA_D_ZERO, MAGMA_D_ONE, dW0_displ, ldda, batchCount, queue ); 


        if ( (n-j-jb) > 0) //update the trailing matrix inside the panel
        {
            magma_dlarft_sm32x32_batched(m-j, jb,
                    dW0_displ, ldda,
                    dW2_displ,
                    dT_array, ldt, 
                    batchCount, queue);

            magma_ddisplace_pointers( dW1_displ, dA_array, ldda, j, j + jb, batchCount, queue );
            magma_dset_pointer( dW2_displ,  dwork, 1, 0, 0,  ldw*n, batchCount, queue );
            magma_dset_pointer( dW3_displ, dwork + ldw*n*batchCount, 1, 0, 0,  ldw*n, batchCount, queue );

            magma_dlarfb_gemm_batched( 
                    MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                    m-j, n-j-jb, jb,
                    (const double**)dW0_displ, ldda,
                    (const double**)dT_array, ldt,
                    dW1_displ,  ldda,
                    dW2_displ,  ldw, 
                    dW3_displ, ldw,
                    batchCount, queue );
        }
    }

    // copy the remaining portion of dR from dA in case m < n
    if ( m < n )
    {
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, 0, minmn, batchCount, queue); 
        magma_ddisplace_pointers(dW3_displ, dR_array, ldr, 0, minmn, batchCount, queue); 
        magmablas_dlacpy_batched( MagmaFull, minmn, n-minmn, dW0_displ, ldda, dW3_displ, ldr, batchCount, queue );
    }
    // to be consistent set the whole upper nbxnb of V to 0/1s, in this case no need to set it inside dgeqrf_batched
    magma_ddisplace_pointers(dW0_displ, dA_array, ldda, 0, 0, batchCount, queue); 
    magmablas_dlaset_batched( MagmaUpper, minmn, n, MAGMA_D_ZERO, MAGMA_D_ONE, dW0_displ, ldda, batchCount, queue ); 


    return MAGMA_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////////////
