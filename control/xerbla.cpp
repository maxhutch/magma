/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    magma_xerbla is an error handler for the MAGMA routines.
    It is called by a MAGMA routine if an input parameter has an
    invalid value. It prints an error message.

    Installers may consider modifying it to
    call system-specific exception-handling facilities.

    Arguments
    ---------
    @param[in]
    srname  CHAR*
            The name of the subroutine that called XERBLA.
            In C/C++ it is convenient to use "__func__".

    @param[in]
    neg_info INTEGER
            Error code.
            Note neg_info's sign is opposite info's normal sign.
            
            Normally:
            - neg_info > 0: The position of the invalid parameter
                         in the parameter list of the calling routine.
            
            The conditions below are also reported, but normally code should not
            call xerbla for these runtime errors:
            - neg_info <  0:          Function-specific error.
            - neg_info >= -MAGMA_ERR: Pre-defined MAGMA error, such as malloc failure.
            - neg_info == 0:          No error.

    @ingroup magma_error
*******************************************************************************/
extern "C"
void magma_xerbla(const char *srname, magma_int_t neg_info)
{
    // the first 3 cases are unusual for calling xerbla;
    // normally runtime errors are passed back in info.
    if ( neg_info < 0 ) {
        fprintf( stderr, "Error in %s, function-specific error (info = %lld)\n",
                 srname, (long long) -neg_info );
    }
    else if ( neg_info == 0 ) {
        fprintf( stderr, "No error, why is %s calling xerbla? (info = %lld)\n",
                 srname, (long long) -neg_info );
    }
    else if ( neg_info >= -MAGMA_ERR ) {
        fprintf( stderr, "Error in %s, %s (info = %lld)\n",
                 srname, magma_strerror(-neg_info), (long long) -neg_info );
    }
    else {
        // this is the normal case for calling xerbla;
        // invalid parameter values are usually logic errors, not runtime errors.
        fprintf( stderr, "On entry to %s, parameter %lld had an illegal value (info = %lld)\n",
                 srname, (long long) neg_info, (long long) -neg_info );
    }
}
