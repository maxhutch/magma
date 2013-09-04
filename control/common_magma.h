/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013
 
       @author Mathieu Faverge
 
       Based on PLASMA common.h
*/

/***************************************************************************//**
 *  MAGMA facilities of interest to both src and magmablas directories
 **/
#ifndef MAGMA_COMMON_H
#define MAGMA_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include <cuda_runtime_api.h>

#if defined( _WIN32 ) || defined( _WIN64 )

    #include "magmawinthread.h"
    #include <windows.h>
    #include <limits.h>
    #include <io.h>

    // functions where Microsoft fails to provide C99 standard
    #define copysign(x,y) _copysign(x,y)
    double log2( double x );  // defined in auxiliary.cpp

#else

    #include <pthread.h>
    #include <unistd.h>
    #include <inttypes.h>

#endif

#if defined(__APPLE__)
    #include "pthread_barrier.h"
#endif

#include "magma.h"
#include "magma_lapack.h"
#include "operators.h"
#include "transpose.h"
#include "magma_threadsetting.h"

/** ****************************************************************************
 * C99 standard defines __func__. Some older compilers use __FUNCTION__.
 * Note __func__ is not a macro, so ifndef __func__ doesn't work.
 */
#if __STDC_VERSION__ < 199901L
# if __GNUC__ >= 2 || _MSC_VER >= 1300
#  define __func__ __FUNCTION__
# else
#  define __func__ "<unknown>"
# endif
#endif

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
 **/
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef roundup
#define roundup(a, b) (b <= 0) ? (a) : (((a) + (b)-1) & ~((b)-1))
#endif

/** ****************************************************************************
 *  Define magma_[sd]sqrt functions 
 *    - sqrt alone cannot be caught by the generation script because of tsqrt
 */

#define magma_dsqrt sqrt
#define magma_ssqrt sqrtf

#endif /* MAGMA_COMMON_H */
