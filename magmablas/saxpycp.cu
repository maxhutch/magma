/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from zaxpycp.cu normal z -> s, Fri Sep 11 18:29:19 2015

*/
#include "common_magma.h"

#define NB 64

// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
saxpycp_kernel(
    int m, float *r, float *x,
    const float *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_S_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


// ----------------------------------------------------------------------
// adds   x += r  --and--
// copies r = b
extern "C" void
magmablas_saxpycp_q(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaFloat_ptr x,
    magmaFloat_const_ptr b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    saxpycp_kernel <<< grid, threads, 0, queue >>> ( m, r, x, b );
}

extern "C" void
magmablas_saxpycp(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaFloat_ptr x,
    magmaFloat_const_ptr b)
{
    magmablas_saxpycp_q( m, r, x, b, magma_stream );
}
