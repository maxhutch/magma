/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
 
       @author Mark Gates
       @generated from cblas_z.cpp normal z -> c, Fri Jan 30 19:00:13 2015

    Wrappers around a few CBLAS functions.
    
    Primarily, we use the standard Fortran BLAS interface in MAGMA. However,
    functions that return a value (as opposed to subroutines that are void)
    are not portable, as they depend on how Fortran returns values. The routines
    here provide a portable interface. These are not identical to CBLAS, in
    particular, [cz]dot[uc] return complex numbers (as in Fortran BLAS) rather
    than return values via an argument.
    
    Only these BLAS-1 functions are provided:
    
    magma_cblas_scasum / dasum
    magma_cblas_scnrm2 / dnrm2
    magma_cblas_cdotc  / ddot
    magma_cblas_cdotu  / ddot

*/
#include <cblas.h>

#include "common_magma.h"

#define COMPLEX

// ========================================
// Level 1 BLAS

// --------------------
/** Returns the sum of absolute values of vector x; i.e., one norm.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    x       COMPLEX array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @ingroup magma_cblas1
*/
extern "C"
float magma_cblas_scasum(
    magma_int_t n,
    const magmaFloatComplex *x, magma_int_t incx )
{
    return cblas_scasum( n, x, incx );
}

// --------------------
/** Returns 2-norm of vector x. Avoids unnecesary over/underflow.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    x       COMPLEX array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @ingroup magma_cblas1
*/
extern "C"
float magma_cblas_scnrm2(
    magma_int_t n,
    const magmaFloatComplex *x, magma_int_t incx )
{
    return cblas_scnrm2( n, x, incx );
}

// --------------------
/** Returns dot product of vectors x and y; \f$ x^H y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    x       COMPLEX array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @param[in]
    y       COMPLEX array on CPU host.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy > 0.

    @ingroup magma_cblas1
*/
extern "C"
magmaFloatComplex magma_cblas_cdotc(
    magma_int_t n,
    const magmaFloatComplex *x, magma_int_t incx,
    const magmaFloatComplex *y, magma_int_t incy )
{
    // after too many issues with MKL and other BLAS, just write our own dot product!
    magmaFloatComplex value = MAGMA_C_ZERO;
    magma_int_t i;
    if ( incx == 1 && incy == 1 ) {
        for( i=0; i < n; ++i ) {
            value += conj( x[i] ) * y[i];
        }
    }
    else {
        magma_int_t ix=0, iy=0;
        if ( incx < 0 ) { ix = (-n + 1)*incx + 1; }
        if ( incy < 0 ) { iy = (-n + 1)*incy + 1; }
        for( magma_int_t i=0; i < n; ++i ) {
            value += conj( x[ix] ) * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return value;
    //#ifdef COMPLEX
    //magmaFloatComplex value;
    //cblas_cdotc_sub( n, x, incx, y, incy, &value );
    //return value;
    //#else
    //return cblas_cdotc( n, x, incx, y, incy );
    //#endif
}

#ifdef COMPLEX
// --------------------
/** Returns dot product (unconjugated) of vectors x and y; \f$ x^T y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    x       COMPLEX array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @param[in]
    y       COMPLEX array on CPU host.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy > 0.

    @ingroup magma_cblas1
*/
extern "C"
magmaFloatComplex magma_cblas_cdotu(
    magma_int_t n,
    const magmaFloatComplex *x, magma_int_t incx,
    const magmaFloatComplex *y, magma_int_t incy )
{
    // after too many issues with MKL and other BLAS, just write our own dot product!
    magmaFloatComplex value = MAGMA_C_ZERO;
    magma_int_t i;
    if ( incx == 1 && incy == 1 ) {
        for( i=0; i < n; ++i ) {
            value += x[i] * y[i];
        }
    }
    else {
        magma_int_t ix=0, iy=0;
        if ( incx < 0 ) { ix = (-n + 1)*incx + 1; }
        if ( incy < 0 ) { iy = (-n + 1)*incy + 1; }
        for( magma_int_t i=0; i < n; ++i ) {
            value += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return value;
    //#ifdef COMPLEX
    //magmaFloatComplex value;
    //cblas_cdotu_sub( n, x, incx, y, incy, &value );
    //return value;
    //#else
    //return cblas_cdotu( n, x, incx, y, incy );
    //#endif
}
#endif

#undef COMPLEX
