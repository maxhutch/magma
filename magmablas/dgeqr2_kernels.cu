/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from zgeqr2_kernels.cu normal z -> d, Fri Jan 30 19:00:10 2015
*/



#include "common_magma.h"
#include "batched_kernel_param.h"



static    double neg_one = MAGMA_D_NEG_ONE;
static    double one  = MAGMA_D_ONE;
static    double zero  = MAGMA_D_ZERO;

__global__ void
dgeqrf_copy_upper_kernel_batched(                
                  int n, int nb,
                  double **dV_array,    int ldv,
                  double **dR_array,    int ldr)
{

    double *dV = dV_array[blockIdx.x];
    double *dR = dR_array[blockIdx.x];

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

void dgeqrf_copy_upper_batched(                
                  magma_int_t n, magma_int_t nb,
                  double **dV_array,    magma_int_t ldv,
                  double **dR_array,    magma_int_t ldr,
          magma_int_t batchCount, magma_queue_t queue)
{
   /* 
        copy some data in dV to dR
   */

      if( nb >= n) return ;

      dgeqrf_copy_upper_kernel_batched<<<batchCount, n, 0, queue>>>(n, nb, dV_array, ldv, dR_array, ldr);

}



extern "C" magma_int_t
magma_dlarfb_dgemm_batched(
                  cublasHandle_t myhandle,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  double **dV_array,    magma_int_t ldv,
                  double **dT_array,    magma_int_t ldt,
                  double **dA_array,    magma_int_t lda,
                  double **W_array,     magma_int_t ldw,
                  double **W2_array,    magma_int_t ldw2,
                  magma_int_t batchCount, magma_queue_t queue)

{

    // W is workspace size of W is nb * n 
    // W = V^H * A. V is stored in A(i:m, i:ib)

    
    if( m <=0 || n <= 0 || k <=0 ) return 1;

#if 1  // CUBLAS is faster than MAGMABLAS by 17GFLOP/S at size 512 batchCount = 2000
    cublasDgemmBatched(myhandle, CUBLAS_OP_C, CUBLAS_OP_N, k, n, m,
                             &one, (const double**) dV_array, ldv,
                                    (const double**) dA_array, lda,
                             &zero,  W_array, ldw, batchCount );



    // W2 = T^H * W        
    cublasDgemmBatched(myhandle, CUBLAS_OP_C, CUBLAS_OP_N, k, n, k,
                             &one, (const double**) dT_array, ldt,
                                    (const double**) W_array, ldw,
                             &zero,  W2_array, ldw2, batchCount );

        
    // A = A - V * W2 
    cublasDgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                             &neg_one, (const double**) dV_array, ldv,
                                    (const double**) W2_array, ldw2,
                             &one,  dA_array, lda, batchCount );

#else 

    magmablas_dgemm_batched(MagmaConjTrans, MagmaNoTrans, k, n, m,
                             one, (const double**) dV_array, ldv,
                                    (const double**) dA_array, lda,
                             zero,  W_array, ldw, batchCount );



    // W2 = T^H * W        
    magmablas_dgemm_batched(MagmaConjTrans, MagmaNoTrans, k, n, k,
                             one, (const double**) dT_array, ldt,
                                    (const double**) W_array, ldw,
                             zero,  W2_array, ldw2, batchCount );

        
    // A = A - V * W2 
    magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, k,
                             neg_one, (const double**) dV_array, ldv,
                                    (const double**) W2_array, ldw2,
                             one,  dA_array, lda, batchCount );
          
#endif       
    return 0;

}



