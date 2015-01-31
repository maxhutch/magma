/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2014
       
       @author Azzam Haidar
       @author Tingxing Dong

       @generated from zgeqrf_panel_batched.cpp normal z -> c, Fri Jan 30 19:00:19 2015
*/
#include "common_magma.h"



////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_cgeqrf_panel_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** tau_array, 
    magmaFloatComplex** dT_array, magma_int_t ldt, 
    magmaFloatComplex** dR_array, magma_int_t ldr,
    magmaFloatComplex** dW0_displ, 
    magmaFloatComplex** dW1_displ,
    magmaFloatComplex   *dwork,  
    magmaFloatComplex** dW2_displ, 
    magmaFloatComplex** dW3_displ,
    magma_int_t *info_array,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{

            magma_int_t j, jb;
            magma_int_t ldw = nb; 

            for( j=0; j<n; j+=nb)
            {
                jb = min(nb, n-j);

                magma_cdisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
                magma_cdisplace_pointers(dW2_displ, tau_array, 1, j, 0, batchCount, queue);
                magma_cdisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue); // 
  
                //sub-panel factorization 
                magma_cgeqr2_batched(
                      m-j, jb,
                      dW0_displ, ldda,      
                      dW2_displ, 
                      info_array, 
                      batchCount,
                      queue);

                //copy upper part of dA to dR
                magma_cdisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
                magma_cdisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue); 
                magmablas_clacpy_batched(MagmaUpper, jb, jb, dW0_displ, ldda, dW3_displ, ldr, batchCount, queue);

                magma_cdisplace_pointers(dW0_displ, dA_array, ldda, j, j, batchCount, queue); 
                magma_cdisplace_pointers(dW3_displ, dR_array, ldr, j, j, batchCount, queue);
                magmablas_claset_batched(MagmaUpper, jb, jb, MAGMA_C_ZERO, MAGMA_C_ONE, dW0_displ, ldda, batchCount, queue); 

                
                if( (n-j-jb) > 0) //update the trailing matrix inside the panel
                {

                    magma_clarft_sm32x32_batched(m-j, jb,
                                 dW0_displ, ldda,
                                 dW2_displ,
                                 dT_array, ldt, 
                                 batchCount, myhandle, queue);

                    magma_cdisplace_pointers(dW1_displ, dA_array, ldda, j, j + jb, batchCount, queue); 
                    cset_pointer(dW2_displ,  dwork, 1, 0, 0,  ldw*n, batchCount, queue );
                    cset_pointer(dW3_displ, dwork + ldw*n*batchCount, 1, 0, 0,  ldw*n, batchCount, queue );    

                    magma_clarfb_gemm_batched(
                                MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                m-j, n-j-jb, jb,
                                (const magmaFloatComplex**)dW0_displ, ldda,
                                (const magmaFloatComplex**)dT_array, ldt,
                                dW1_displ,  ldda,
                                dW2_displ,  ldw, 
                                dW3_displ, ldw, batchCount, myhandle, queue);
                }
               
            }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////

