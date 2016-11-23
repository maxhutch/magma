/*
   -- MAGMA (version 2.2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date November 2016

   @author Azzam Haidar
   @author Ahmad Abdelfattah

   @generated from magmablas/zpotf2_kernels_var.cu, normal z -> d, Sun Nov 20 20:20:32 2016
 */
#define PRECISION_d

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"

//#define VBATCH_DISABLE_THREAD_RETURN
#ifdef VBATCH_DISABLE_THREAD_RETURN
#define ENABLE_COND1
#define ENABLE_COND2
#define ENABLE_COND4
#define ENABLE_COND5
#define ENABLE_COND6
#endif

#define MAX_NTCOL 1
#include "dpotf2_devicesfunc.cuh"
/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dpotf2_smlpout_kernel_vbatched_v2(int maxm, magma_int_t *m, 
        double **dA_array, magma_int_t *lda, 
        int localstep, int gbstep, magma_int_t *info_array)
{
    const int batchid   = blockIdx.z;
    const int my_m      = (int)m[batchid];
    const int mylda     = (int)lda[batchid];

    const int myoff     = ((maxm - my_m)/POTF2_NB)*POTF2_NB;
    const int mylocstep = localstep - myoff;
    const int myrows    = mylocstep >= 0 ? my_m-mylocstep : 0;
    const int myib      = min(POTF2_NB, myrows);

    #ifndef VBATCH_DISABLE_THREAD_RETURN
    const int tx = threadIdx.x; 
    if(tx >=  myrows) return;
    #else
    if(myrows <= 0) return;   
    #endif
    
    if(myib == POTF2_NB)
        dpotf2_smlpout_fixwidth_device( myrows, dA_array[batchid]+mylocstep, dA_array[batchid]+mylocstep+mylocstep*mylda, mylda, mylocstep, gbstep, &(info_array[batchid]));
    else
        dpotf2_smlpout_anywidth_device( myrows, myib, dA_array[batchid]+mylocstep, dA_array[batchid]+mylocstep+mylocstep*mylda, mylda, mylocstep, gbstep, &(info_array[batchid]));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dpotf2_smlpout_kernel_vbatched(magma_int_t *m, 
        double **dA_array, magma_int_t *lda, 
        int localstep, int gbstep, magma_int_t *info_array)
{
    const int batchid = blockIdx.z;
    const int myrows  = (int)m[batchid] - localstep;
    const int myib    = min(POTF2_NB, myrows);
    const int mylda   = lda[batchid];
    
    #ifndef VBATCH_DISABLE_THREAD_RETURN
    const int tx = threadIdx.x; 
    if(tx >=  myrows) return; 
    #else
    if(myrows <= 0) return; 
    #endif
    
    if(myib == POTF2_NB)
        dpotf2_smlpout_fixwidth_device( myrows, dA_array[batchid]+localstep, dA_array[batchid]+localstep+localstep*mylda, mylda, localstep, gbstep, &(info_array[batchid]));
    else
        dpotf2_smlpout_anywidth_device( myrows, myib, dA_array[batchid]+localstep, dA_array[batchid]+localstep+localstep*mylda, mylda, localstep, gbstep, &(info_array[batchid]));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dpotrf_lpout_vbatched(
        magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,  
        double **dA_array, magma_int_t *lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    // Quick return if possible
    if (max_n <= 0) {
        arginfo = -33;  // any value for now
        return arginfo;
    }

    dim3 dimGrid(1, 1, batchCount);
    for(magma_int_t j = 0; j < max_n; j+= POTF2_NB) {
        magma_int_t rows_max = max_n-j;
        magma_int_t nbth = rows_max; 
        dim3 threads(nbth, 1);
        magma_int_t shared_mem_size = sizeof(double)*(nbth+POTF2_NB)*POTF2_NB;
        if(shared_mem_size > 47000) 
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }
        //dpotf2_smlpout_kernel_vbatched<<<dimGrid, threads, shared_mem_size, queue >>>(n, dA_array, lda, j, gbstep, info_array);
        dpotf2_smlpout_kernel_vbatched_v2
        <<<dimGrid, threads, shared_mem_size, queue->cuda_stream() >>>
        (max_n, n, dA_array, lda, j, gbstep, info_array);
    }
    return arginfo;
}
