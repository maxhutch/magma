/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/



#include "common_magma.h"
#include "batched_kernel_param.h"



static    magmaDoubleComplex neg_one = MAGMA_Z_NEG_ONE;
static    magmaDoubleComplex one  = MAGMA_Z_ONE;
static    magmaDoubleComplex zero  = MAGMA_Z_ZERO;

__global__ void
zgeqrf_copy_upper_kernel_batched(                
                  int n, int nb,
                  magmaDoubleComplex **dV_array,    int ldv,
                  magmaDoubleComplex **dR_array,    int ldr)
{

    magmaDoubleComplex *dV = dV_array[blockIdx.x];
    magmaDoubleComplex *dR = dR_array[blockIdx.x];

    int tid = threadIdx.x;

    int column = (tid / nb + 1) * nb; 
    
    if( tid < n && column < n) 
    {
       for(int i=column; i<n; i++)
       {
          dR[tid + i * ldr]  =  dV[tid + i * ldv];  
       }
    }
}

void zgeqrf_copy_upper_batched(                
                  magma_int_t n, magma_int_t nb,
                  magmaDoubleComplex **dV_array,    magma_int_t ldv,
                  magmaDoubleComplex **dR_array,    magma_int_t ldr,
          magma_int_t batchCount, magma_queue_t queue)
{
   /* 
        copy some data in dV to dR
   */

      if( nb >= n) return ;

      zgeqrf_copy_upper_kernel_batched<<<batchCount, n, 0, queue>>>(n, nb, dV_array, ldv, dR_array, ldr);

}



extern "C" magma_int_t
magma_zlarfb_zgemm_batched(
                  cublasHandle_t myhandle,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  magmaDoubleComplex **dV_array,    magma_int_t ldv,
                  magmaDoubleComplex **dT_array,    magma_int_t ldt,
                  magmaDoubleComplex **dA_array,    magma_int_t lda,
                  magmaDoubleComplex **W_array,     magma_int_t ldw,
                  magmaDoubleComplex **W2_array,    magma_int_t ldw2,
                  magma_int_t batchCount, magma_queue_t queue)

{

    // W is workspace size of W is nb * n 
    // W = V^H * A. V is stored in A(i:m, i:ib)

    
    if( m <=0 || n <= 0 || k <=0 ) return 1;

#if 1  // CUBLAS is faster than MAGMABLAS by 17GFLOP/S at size 512 batchCount = 2000
    cublasZgemmBatched(myhandle, CUBLAS_OP_C, CUBLAS_OP_N, k, n, m,
                             &one, (const magmaDoubleComplex**) dV_array, ldv,
                                    (const magmaDoubleComplex**) dA_array, lda,
                             &zero,  W_array, ldw, batchCount );



    // W2 = T^H * W        
    cublasZgemmBatched(myhandle, CUBLAS_OP_C, CUBLAS_OP_N, k, n, k,
                             &one, (const magmaDoubleComplex**) dT_array, ldt,
                                    (const magmaDoubleComplex**) W_array, ldw,
                             &zero,  W2_array, ldw2, batchCount );

        
    // A = A - V * W2 
    cublasZgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                             &neg_one, (const magmaDoubleComplex**) dV_array, ldv,
                                    (const magmaDoubleComplex**) W2_array, ldw2,
                             &one,  dA_array, lda, batchCount );

#else 

    magmablas_zgemm_batched(MagmaConjTrans, MagmaNoTrans, k, n, m,
                             one, (const magmaDoubleComplex**) dV_array, ldv,
                                    (const magmaDoubleComplex**) dA_array, lda,
                             zero,  W_array, ldw, batchCount );



    // W2 = T^H * W        
    magmablas_zgemm_batched(MagmaConjTrans, MagmaNoTrans, k, n, k,
                             one, (const magmaDoubleComplex**) dT_array, ldt,
                                    (const magmaDoubleComplex**) W_array, ldw,
                             zero,  W2_array, ldw2, batchCount );

        
    // A = A - V * W2 
    magmablas_zgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, k,
                             neg_one, (const magmaDoubleComplex**) dV_array, ldv,
                                    (const magmaDoubleComplex**) W2_array, ldw2,
                             one,  dA_array, lda, batchCount );
          
#endif       
    return 0;

}



