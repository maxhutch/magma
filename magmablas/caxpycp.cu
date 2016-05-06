/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zaxpycp.cu normal z -> c, Mon May  2 23:30:29 2016

*/
#include "magma_internal.h"

#define NB 64

// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
caxpycp_kernel(
    int m,
    magmaFloatComplex *r,
    magmaFloatComplex *x,
    const magmaFloatComplex *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_C_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


// ----------------------------------------------------------------------
// adds   x += r  --and--
// copies r = b
extern "C" void
magmablas_caxpycp_q(
    magma_int_t m,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_const_ptr b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    caxpycp_kernel <<< grid, threads, 0, queue->cuda_stream() >>> ( m, r, x, b );
}
