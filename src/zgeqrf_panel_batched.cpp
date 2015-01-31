/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2014
       
       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/
#include "common_magma.h"



////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgeqrf_panel_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** tau_array, 
    magmaDoubleComplex** dT_array, magma_int_t ldt, 
    magmaDoubleComplex** dR_array, magma_int_t ldr,
    magmaDoubleComplex** dW0_displ, 
    magmaDoubleComplex** dW1_displ,
    magmaDoubleComplex   *dwork,  
    magmaDoubleComplex** dW2_displ, 
    magmaDoubleComplex** dW3_displ,
    magma_int_t *info_array,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{

            magma_int_t j, jb;
            magma_int_t ldw = nb; 

            for( j=0; j<n; j+=nb)
            {
                jb = min(nb, n-j);

                magma_zdisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
                magma_zdisplace_pointers(dW2_displ, tau_array, 1, j, 0, batchCount, queue);
                magma_zdisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue); // 
  
                //sub-panel factorization 
                magma_zgeqr2_batched(
                      m-j, jb,
                      dW0_displ, ldda,      
                      dW2_displ, 
                      info_array, 
                      batchCount,
                      queue);

                //copy upper part of dA to dR
                magma_zdisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
                magma_zdisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue); 
                magmablas_zlacpy_batched(MagmaUpper, jb, jb, dW0_displ, ldda, dW3_displ, ldr, batchCount, queue);

                magma_zdisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
                magma_zdisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue);
                magmablas_zlaset_batched(MagmaUpper, jb, jb, MAGMA_Z_ZERO, MAGMA_Z_ONE, dW0_displ, ldda, batchCount, queue); 

                
                if( (n-j-jb) > 0) //update the trailing matrix inside the panel
                {

                    magma_zlarft_sm32x32_batched(m-j, jb,
                                 dW0_displ, ldda,
                                 dW2_displ,
                                 dT_array, ldt, 
                                 batchCount, myhandle, queue);

                    magma_zdisplace_pointers(dW1_displ, dA_array, ldda, j, j + jb, batchCount, queue); 
                    zset_pointer(dW2_displ,  dwork, 1, 0, 0,  ldw*n, batchCount, queue );
                    zset_pointer(dW3_displ, dwork + ldw*n*batchCount, 1, 0, 0,  ldw*n, batchCount, queue );    

                    magma_zlarfb_gemm_batched(
                                MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                m-j, n-j-jb, jb,
                                (const magmaDoubleComplex**)dW0_displ, ldda,
                                (const magmaDoubleComplex**)dT_array, ldt,
                                dW1_displ,  ldda,
                                dW2_displ,  ldw, 
                                dW3_displ, ldw, batchCount, myhandle, queue);
                }
               
            }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////

