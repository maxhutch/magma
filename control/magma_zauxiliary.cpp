/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates
       @precisions normal z -> s d c
*/
#include "magma_internal.h"

#define PRECISION_z

/***************************************************************************//**
    Purpose
    -------
    This deals with a subtle bug with returning lwork as a Float.
    If lwork > 2**24, then it will get rounded as a Float;
    we need to ensure it is rounded up instead of down,
    by multiplying by 1.+eps in Double precision:

        float( 16777217            ) == 16777216
        float( 16777217 * (1.+eps) ) == 16777218

    where eps is Single precision machine epsilon.
    (Could use 1+2*eps in Single precision, but that can add more than necesary.)
    If lwork > 2**53, rounding would happen in Double, too, but that's 94M x 94M!

    @param[in]
    lwork   Workspace size.

    @return lwork, converted to magmaDoubleComplex and rounded up slightly
            if necesary so that returned lwork >= input lwork.

    @ingroup magma_make_lwork
*******************************************************************************/
magmaDoubleComplex magma_zmake_lwork( magma_int_t lwork )
{
    #if defined(PRECISION_s) || defined(PRECISION_c)
    real_Double_t one_eps = 1. + lapackf77_dlamch("Epsilon");
    return MAGMA_Z_MAKE( double(lwork*one_eps), 0 );
    #else
    return MAGMA_Z_MAKE( double(lwork), 0 );
    #endif
}
