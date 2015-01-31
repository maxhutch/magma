#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "common_magma.h"

extern "C"
void magma_xerbla(const char *srname , magma_int_t info)
{
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

    Purpose
    =======

    magma_xerbla is an error handler for the MAGMA routines.
    It is called by a MAGMA routine if an input parameter has an
    invalid value. It calls the LAPACK XERBLA routine, which by default
    prints an error message and stops execution.

    Installers may consider modifying the STOP statement in order to
    call system-specific exception-handling facilities.

    Arguments
    =========

    SRNAME  (input) CHARACTER*(*)
            The name of the routine which called XERBLA.
            In C it is convenient to use __func__.

    INFO    (input) INTEGER
            The position of the invalid parameter in the parameter list
            of the calling routine.

    =====================================================================   */

    int len = strlen( srname );
    lapackf77_xerbla( srname, &info, len );
}
