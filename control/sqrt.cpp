/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
*/

#include <complex>

#include "magma_operators.h"

/**
    @return Complex square root of x.
    
    @param[in]
    x       COMPLEX_16
    
    @ingroup magma_zaux0
    ********************************************************************/
magmaDoubleComplex magma_zsqrt( magmaDoubleComplex x )
{
    std::complex<double> y = std::sqrt( std::complex<double>( real(x), imag(x) ));
    return MAGMA_Z_MAKE( real(y), imag(y) );
}


/**
    @return Complex square root of x.
    
    @param[in]
    x       COMPLEX
    
    @ingroup magma_caux0
    ********************************************************************/
magmaFloatComplex magma_csqrt( magmaFloatComplex x )
{
    std::complex<float> y = std::sqrt( std::complex<float>( real(x), imag(x) ));
    return MAGMA_C_MAKE( real(y), imag(y) );
}
