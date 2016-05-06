/*
   -- MAGMA (version 2.0.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date May 2016

   @author Azzam Haidar
   @author Ahmad Ahmad

   @generated from magmablas/zpotf2_kernels.cu normal z -> s, Mon May  2 23:30:41 2016
 */
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_s



#if defined(VERSION31)
    #define ENABLE_COND1
    #define ENABLE_COND2
#endif

#define MAX_NTCOL 8
#ifdef PRECISION_s
#define NTCOL2   (4)
#define NTCOL1   (8)
#elif defined(PRECISION_d)
#define NTCOL2   (2)
#define NTCOL1   (4)
#else
#define NTCOL2   (1)
#define NTCOL1   (1)
#endif


#include "spotf2_devicesfunc.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void spotf2_smlpout_fixwidth_kernel_batched(int m, 
        float **dA_array, int lda, 
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.z * blockDim.y + threadIdx.y;
    if (batchid >= batchCount) return;
    spotf2_smlpout_fixwidth_device(m, dA_array[batchid]+localstep, dA_array[batchid]+localstep+localstep*lda, lda, localstep, gbstep, &(info_array[batchid]));
}


/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void spotf2_smlpout_anywidth_kernel_batched(int m, int n, 
        float **dA_array, int lda, 
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.z * blockDim.y + threadIdx.y;
    if (batchid >= batchCount) return;
    spotf2_smlpout_anywidth_device(m, n, dA_array[batchid]+localstep, dA_array[batchid]+localstep+localstep*lda, lda, localstep, gbstep, &(info_array[batchid]));
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
/**
    This is an internal routine.
    ********************************************************************/
extern "C" magma_int_t
magma_spotrf_lpout_batched(
        magma_uplo_t uplo, magma_int_t n, 
        float **dA_array, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t roundup_m = m;
    // rounding up need more investigation since it coul dmodify the matrix out of its bound
    //magma_int_t m8  = magma_roundup( m, 8 );
    //magma_int_t roundup_m = m8 > lda ? m : m8;
    //magma_int_t m32 = magma_roundup( m, 32 );
    //magma_int_t roundup_m = m32 > lda ? m : m32;

    magma_int_t  ib, rows;
    
    for (magma_int_t j = 0; j < n; j += POTF2_NB) {
        ib   = min(POTF2_NB, n-j);
        rows = roundup_m-j;
        
        // tuning ntcol
        magma_int_t ntcol;  // for z precision, the best tuning is at NTCOL = 1 for all sizes
        if (rows > 64) ntcol = 1;
        else if (rows > 32) ntcol = NTCOL2;
        else ntcol = NTCOL1;
        // end of tuning ntcol
        
        const magma_int_t nTB = magma_ceildiv( batchCount, ntcol );
        dim3 dimGrid(1, 1, nTB);
        magma_int_t nbth = rows;
        magma_int_t shared_mem_size = ntcol * (sizeof(float)*(nbth+POTF2_NB)*POTF2_NB);
        dim3 threads(nbth, ntcol);
        
        if (shared_mem_size > 47000) 
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }

        if (ib == POTF2_NB)
        {
            spotf2_smlpout_fixwidth_kernel_batched
                <<< dimGrid, threads, shared_mem_size, queue->cuda_stream() >>>
                (rows, dA_array, lda, j, gbstep, info_array, batchCount);
        } else {
            spotf2_smlpout_anywidth_kernel_batched
                <<< dimGrid, threads, shared_mem_size, queue->cuda_stream() >>>
                (rows, ib, dA_array, lda, j, gbstep, info_array, batchCount);
        }
    }

    return arginfo;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
