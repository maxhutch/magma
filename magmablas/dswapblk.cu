/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zswapblk.cu normal z -> d, Mon May  2 23:30:35 2016

*/
#include "magma_internal.h"

#define BLOCK_SIZE 64

/*********************************************************/
/*
 *  Blocked version: swap several pairs of lines
 */
typedef struct {
    double *A;
    double *B;
    int n, ldda, lddb, npivots;
    short ipiv[BLOCK_SIZE];
} magmagpu_dswapblk_params_t;

__global__ void magmagpu_dswapblkrm( magmagpu_dswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    if ( y < params.n )
    {
        double *A = params.A + y - params.ldda;
        double *B = params.B + y;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A += params.ldda;
            if ( params.ipiv[i] == -1 )
                continue;
            double  tmp1 = *A;
            double *tmp2 = B + params.ipiv[i]*params.lddb;
            *A    = *tmp2;
            *tmp2 =  tmp1;
        }
    }
}

__global__ void magmagpu_dswapblkcm( magmagpu_dswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int offset1 = y*params.ldda;
    unsigned int offset2 = y*params.lddb;
    if ( y < params.n )
    {
        double *A = params.A + offset1 - 1;
        double *B = params.B + offset2;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A++;
            if ( params.ipiv[i] == -1 )
                continue;
            double  tmp1 = *A;
            double *tmp2 = B + params.ipiv[i];
            *A    = *tmp2;
            *tmp2 =  tmp1;
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
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset,
    magma_queue_t queue )
{
    magma_int_t  blocksize = 64;
    dim3 blocks( magma_ceildiv( n, blocksize ) );
    magma_int_t  k, im;
    
    /* Quick return */
    if ( n == 0 )
        return;
    
    if ( order == MagmaColMajor ) {
        for( k=(i1-1); k < i2; k += BLOCK_SIZE )
        {
            magma_int_t sb = min(BLOCK_SIZE, i2-k);
            magmagpu_dswapblk_params_t params = { dA+k, dB, int(n), int(ldda), int(lddb), int(sb) };
            for( magma_int_t j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im )
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            magmagpu_dswapblkcm<<< blocks, blocksize, 0, queue->cuda_stream() >>>( params );
        }
    }
    else {
        for( k=(i1-1); k < i2; k += BLOCK_SIZE )
        {
            magma_int_t sb = min(BLOCK_SIZE, i2-k);
            magmagpu_dswapblk_params_t params = { dA+k*ldda, dB, int(n), int(ldda), int(lddb), int(sb) };
            for( magma_int_t j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im )
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            magmagpu_dswapblkrm<<< blocks, blocksize, 0, queue->cuda_stream() >>>( params );
        }
    }
}
