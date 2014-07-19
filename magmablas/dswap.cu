/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zswap.cu normal z -> d, Fri Jul 18 17:34:12 2014

*/
#include "common_magma.h"

#define BLOCK_SIZE 64

/*********************************************************
 *
 * SWAP BLAS: permute to set of N elements
 *
 ********************************************************/
/*
 *  First version: line per line
 */
typedef struct {
    double *A1;
    double *A2;
    int n, lda1, lda2;
} magmagpu_dswap_params_t;

__global__ void magmagpu_dswap( magmagpu_dswap_params_t params )
{
    unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int offset1 = x*params.lda1;
    unsigned int offset2 = x*params.lda2;
    if( x < params.n )
    {
        double *A1  = params.A1 + offset1;
        double *A2  = params.A2 + offset2;
        double temp = *A1;
        *A1 = *A2;
        *A2 = temp;
    }
}

extern "C" void 
magmablas_dswap( magma_int_t n, double *dA1T, magma_int_t lda1, 
                 double *dA2T, magma_int_t lda2)
{
    int blocksize = 64;
    dim3 blocks( (n+blocksize-1) / blocksize, 1, 1);
    magmagpu_dswap_params_t params = { dA1T, dA2T, n, lda1, lda2 };
    magmagpu_dswap<<< blocks, blocksize, 0, magma_stream >>>( params );
}

