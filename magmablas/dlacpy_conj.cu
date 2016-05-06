/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zlacpy_conj.cu normal z -> d, Mon May  2 23:30:31 2016

*/
#include "magma_internal.h"

#define BLOCK_SIZE 64

// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old dswap routine. Update to new dlacpy code for 2D matrix?

__global__ void dlacpy_conj_kernel(
    int n,
    double *A1, int lda1,
    double *A2, int lda2 )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_D_CONJ( A1[offset1] );
    }
}


extern "C" void 
magmablas_dlacpy_conj_q(
    magma_int_t n,
    magmaDouble_ptr dA1, magma_int_t lda1, 
    magmaDouble_ptr dA2, magma_int_t lda2,
    magma_queue_t queue )
{
    dim3 threads( BLOCK_SIZE );
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ) );
    dlacpy_conj_kernel<<< blocks, threads, 0, queue->cuda_stream() >>>( n, dA1, lda1, dA2, lda2 );
}
