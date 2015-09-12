/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from zaxpycp.cu normal z -> d, Fri Sep 11 18:29:19 2015

*/
#include "common_magma.h"

#define NB 64

// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
daxpycp_kernel(
    int m, double *r, double *x,
    const double *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_D_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


// ----------------------------------------------------------------------
// adds   x += r  --and--
// copies r = b
extern "C" void
magmablas_daxpycp_q(
    magma_int_t m,
    magmaDouble_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    daxpycp_kernel <<< grid, threads, 0, queue >>> ( m, r, x, b );
}

extern "C" void
magmablas_daxpycp(
    magma_int_t m,
    magmaDouble_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b)
{
    magmablas_daxpycp_q( m, r, x, b, magma_stream );
}
