/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from zgeqr2_kernels.cu normal z -> c, Fri Jan 30 19:00:10 2015
*/



#include "common_magma.h"
#include "batched_kernel_param.h"



static    magmaFloatComplex neg_one = MAGMA_C_NEG_ONE;
static    magmaFloatComplex one  = MAGMA_C_ONE;
static    magmaFloatComplex zero  = MAGMA_C_ZERO;

__global__ void
cgeqrf_copy_upper_kernel_batched(                
                  int n, int nb,
                  magmaFloatComplex **dV_array,    int ldv,
                  magmaFloatComplex **dR_array,    int ldr)
{

    magmaFloatComplex *dV = dV_array[blockIdx.x];
    magmaFloatComplex *dR = dR_array[blockIdx.x];

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

void cgeqrf_copy_upper_batched(                
                  magma_int_t n, magma_int_t nb,
                  magmaFloatComplex **dV_array,    magma_int_t ldv,
                  magmaFloatComplex **dR_array,    magma_int_t ldr,
          magma_int_t batchCount, magma_queue_t queue)
{
   /* 
        copy some data in dV to dR
   */

      if( nb >= n) return ;

      cgeqrf_copy_upper_kernel_batched<<<batchCount, n, 0, queue>>>(n, nb, dV_array, ldv, dR_array, ldr);

}



extern "C" magma_int_t
magma_clarfb_cgemm_batched(
                  cublasHandle_t myhandle,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  magmaFloatComplex **dV_array,    magma_int_t ldv,
                  magmaFloatComplex **dT_array,    magma_int_t ldt,
                  magmaFloatComplex **dA_array,    magma_int_t lda,
                  magmaFloatComplex **W_array,     magma_int_t ldw,
                  magmaFloatComplex **W2_array,    magma_int_t ldw2,
                  magma_int_t batchCount, magma_queue_t queue)

{

    // W is workspace size of W is nb * n 
    // W = V^H * A. V is stored in A(i:m, i:ib)

    
    if( m <=0 || n <= 0 || k <=0 ) return 1;

#if 1  // CUBLAS is faster than MAGMABLAS by 17GFLOP/S at size 512 batchCount = 2000
    cublasCgemmBatched(myhandle, CUBLAS_OP_C, CUBLAS_OP_N, k, n, m,
                             &one, (const magmaFloatComplex**) dV_array, ldv,
                                    (const magmaFloatComplex**) dA_array, lda,
                             &zero,  W_array, ldw, batchCount );



    // W2 = T^H * W        
    cublasCgemmBatched(myhandle, CUBLAS_OP_C, CUBLAS_OP_N, k, n, k,
                             &one, (const magmaFloatComplex**) dT_array, ldt,
                                    (const magmaFloatComplex**) W_array, ldw,
                             &zero,  W2_array, ldw2, batchCount );

        
    // A = A - V * W2 
    cublasCgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                             &neg_one, (const magmaFloatComplex**) dV_array, ldv,
                                    (const magmaFloatComplex**) W2_array, ldw2,
                             &one,  dA_array, lda, batchCount );

#else 

    magmablas_cgemm_batched(MagmaConjTrans, MagmaNoTrans, k, n, m,
                             one, (const magmaFloatComplex**) dV_array, ldv,
                                    (const magmaFloatComplex**) dA_array, lda,
                             zero,  W_array, ldw, batchCount );



    // W2 = T^H * W        
    magmablas_cgemm_batched(MagmaConjTrans, MagmaNoTrans, k, n, k,
                             one, (const magmaFloatComplex**) dT_array, ldt,
                                    (const magmaFloatComplex**) W_array, ldw,
                             zero,  W2_array, ldw2, batchCount );

        
    // A = A - V * W2 
    magmablas_cgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, k,
                             neg_one, (const magmaFloatComplex**) dV_array, ldv,
                                    (const magmaFloatComplex**) W2_array, ldw2,
                             one,  dA_array, lda, batchCount );
          
#endif       
    return 0;

}



