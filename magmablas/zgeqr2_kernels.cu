/*
   -- MAGMA (version 1.6.3-beta1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date August 2015

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
#include "common_magma.h"
#include "batched_kernel_param.h"


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
    
    if ( tid < n && column < n) 
    {
        for (int i=column; i < n; i++)
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
    if ( nb >= n) return;
    
    zgeqrf_copy_upper_kernel_batched<<<batchCount, n, 0, queue>>>(n, nb, dV_array, ldv, dR_array, ldr);
}
