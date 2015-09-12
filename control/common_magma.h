/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @author Mathieu Faverge
       @author Mark Gates

       Based on PLASMA common.h
*/

/***************************************************************************//**
 *  MAGMA facilities of interest to both src and magmablas directories
 **/
#ifndef MAGMA_COMMON_H
#define MAGMA_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#if defined( _WIN32 ) || defined( _WIN64 )

    #include "magma_winthread.h"
    #include <windows.h>
    #include <limits.h>
    #include <io.h>

    // functions where Microsoft fails to provide C99 standard
    // (only with Microsoft, not with nvcc on Windows)
    // in both common_magma.h and testings.h
    #ifndef __NVCC__

        #include <float.h>
        #define copysign(x,y) _copysign(x,y)
        #define isnan(x)      _isnan(x)
        #define isinf(x)      ( ! _finite(x) && ! _isnan(x) )
        #define isfinite(x)   _finite(x)
        // note _snprintf has slightly different semantics than snprintf
        #define snprintf _snprintf

    #endif

#else

    #include <pthread.h>
    #include <unistd.h>
    #include <inttypes.h>

#endif

// provide our own support for pthread_barrier on MacOS and Windows
#include "pthread_barrier.h"

#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "transpose.h"
#include "magma_threadsetting.h"

/** ****************************************************************************
 *  Determine if weak symbols are allowed
 */
#if defined(linux) || defined(__linux) || defined(__linux__)
#if defined(__GNUC_EXCL__) || defined(__GNUC__)
#define MAGMA_HAVE_WEAK    1
#endif
#endif

/***************************************************************************//**
 *  Global utilities
 *  in both common_magma.h and testings.h
 **/
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// suppress "warning: unused variable" in a portable fashion
#define MAGMA_UNUSED(var)  ((void)var)


/** ****************************************************************************
 *  Internal global variables
 */
extern magma_queue_t magma_stream;

#endif /* MAGMA_COMMON_H */
