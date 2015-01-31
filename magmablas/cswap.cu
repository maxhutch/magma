/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Mark Gates

       @generated from zswap.cu normal z -> c, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"

#define NB 64


/* Vector is divided into ceil(n/nb) blocks.
   Each thread swaps one element, x[tid] <---> y[tid].
*/
__global__ void cswap_kernel(
    int n,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex *y, int incy )
{
    magmaFloatComplex tmp;
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    if ( ind < n ) {
        x += ind*incx;
        y += ind*incy;
        tmp = *x;
        *x  = *y;
        *y  = tmp;
    }
}


/**
    Purpose:
    =============
    Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_cblas1
    ********************************************************************/
extern "C" void 
magmablas_cswap_q(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx, 
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    dim3 grid( (n+NB-1) / NB );
    dim3 threads( NB );
    cswap_kernel<<< grid, threads, 0, queue >>>( n, dx, incx, dy, incy );
}


/**
    @see magmablas_cswap_q
    @ingroup magma_cblas1
    ********************************************************************/
extern "C" void 
magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx, 
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cswap_q( n, dx, incx, dy, incy, magma_stream );
}
