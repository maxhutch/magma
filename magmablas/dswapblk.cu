/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from zswapblk.cu normal z -> d, Tue Sep  2 12:38:16 2014

*/
#include "common_magma.h"

#define BLOCK_SIZE 64

/*********************************************************/
/*
 *  Blocked version: swap several pairs of lines
 */
typedef struct {
    double *A1;
    double *A2;
    int n, lda1, lda2, npivots;
    short ipiv[BLOCK_SIZE];
} magmagpu_dswapblk_params_t;

__global__ void magmagpu_dswapblkrm( magmagpu_dswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    if( y < params.n )
    {
        double *A1 = params.A1 + y - params.lda1;
        double *A2 = params.A2 + y;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A1 += params.lda1;
            if ( params.ipiv[i] == -1 )
                continue;
            double tmp1  = *A1;
            double *tmp2 = A2 + params.ipiv[i]*params.lda2;
            *A1   = *tmp2;
            *tmp2 = tmp1;
        }
    }
}

__global__ void magmagpu_dswapblkcm( magmagpu_dswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int offset1 = y*params.lda1;
    unsigned int offset2 = y*params.lda2;
    if( y < params.n )
    {
        double *A1 = params.A1 + offset1 - 1;
        double *A2 = params.A2 + offset2;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A1++;
            if ( params.ipiv[i] == -1 )
                continue;
            double tmp1  = *A1;
            double *tmp2 = A2 + params.ipiv[i];
            *A1   = *tmp2;
            *tmp2 = tmp1;
        }
    }
    __syncthreads();
}


/**
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void 
magmablas_dswapblk_q(
    magma_order_t order, magma_int_t n, 
    double *dA1T, magma_int_t lda1,
    double *dA2T, magma_int_t lda2,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset,
    magma_queue_t queue )
{
    magma_int_t  blocksize = 64;
    dim3 blocks( (n+blocksize-1) / blocksize, 1, 1);
    magma_int_t  k, im;
    
    /* Quick return */
    if ( n == 0 )
        return;
    
    if ( order == MagmaColMajor ) {
        for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
        {
            magma_int_t sb = min(BLOCK_SIZE, i2-k);
            magmagpu_dswapblk_params_t params = { dA1T+k, dA2T, n, lda1, lda2, sb };
            for( magma_int_t j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im )
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            magmagpu_dswapblkcm<<< blocks, blocksize, 0, queue >>>( params );
        }
    }
    else {
        for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
        {
            magma_int_t sb = min(BLOCK_SIZE, i2-k);
            magmagpu_dswapblk_params_t params = { dA1T+k*lda1, dA2T, n, lda1, lda2, sb };
            for( magma_int_t j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im )
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            magmagpu_dswapblkrm<<< blocks, blocksize, 0, queue >>>( params );
        }
    }
}


/**
    @see magmablas_dswapblk_q
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void 
magmablas_dswapblk(
    magma_order_t order, magma_int_t n, 
    double *dA1T, magma_int_t lda1,
    double *dA2T, magma_int_t lda2,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_dswapblk_q(
        order, n, dA1T, lda1, dA2T, lda2, i1, i2, ipiv, inci, offset, magma_stream );
}
