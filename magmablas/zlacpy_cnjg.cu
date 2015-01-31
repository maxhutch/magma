/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

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
    magmaDoubleComplex *A1;
    magmaDoubleComplex *A2;
    int n, lda1, lda2;
} magmagpu_zlacpy_cnjg_params_t;

__global__ void magmagpu_zlacpy_cnjg( magmagpu_zlacpy_cnjg_params_t params )
{
    unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int offset1 = x*params.lda1;
    unsigned int offset2 = x*params.lda2;
    if( x < params.n )
    {
        magmaDoubleComplex *A1  = params.A1 + offset1;
        magmaDoubleComplex *A2  = params.A2 + offset2;
        *A2 = MAGMA_Z_CNJG(*A1);
    }
}


extern "C" void 
magmablas_zlacpy_cnjg_q(
    magma_int_t n, magmaDoubleComplex *dA1, magma_int_t lda1, 
    magmaDoubleComplex *dA2, magma_int_t lda2,
    magma_queue_t queue )
{
    int blocksize = 64;
    dim3 blocks( (n+blocksize-1) / blocksize, 1, 1);
    magmagpu_zlacpy_cnjg_params_t params = { dA1, dA2, n, lda1, lda2 };
    magmagpu_zlacpy_cnjg<<< blocks, blocksize, 0, queue >>>( params );
}


extern "C" void 
magmablas_zlacpy_cnjg(
    magma_int_t n, magmaDoubleComplex *dA1, magma_int_t lda1, 
    magmaDoubleComplex *dA2, magma_int_t lda2)
{
    magmablas_zlacpy_cnjg_q( n, dA1, lda1, dA2, lda2, magma_stream );
}
