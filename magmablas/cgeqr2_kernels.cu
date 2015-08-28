/*
   -- MAGMA (version 1.6.3-beta1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date August 2015

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from zgeqr2_kernels.cu normal z -> c, Tue Aug 25 16:35:10 2015
*/
#include "common_magma.h"
#include "batched_kernel_param.h"


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
    
    if ( tid < n && column < n) 
    {
        for (int i=column; i < n; i++)
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
    if ( nb >= n) return;
    
    cgeqrf_copy_upper_kernel_batched<<<batchCount, n, 0, queue>>>(n, nb, dV_array, ldv, dR_array, ldr);
}
