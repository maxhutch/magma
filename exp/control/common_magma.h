/**
 *
 * @file common_magma.h
 *
 *  MAGMA (version 1.6.1) --
 *  Univ. of Tennessee, Knoxville
 *  Univ. of California, Berkeley
 *  Univ. of Colorado, Denver
 *  @date January 2015
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date January 2015
 *
 * Based on PLASMA common.h
 *
 **/

/***************************************************************************//**
 *  MAGMA facilities of interest to both src and magmablas directories
 **/
#ifndef _MAGMA_COMMON_H_
#define _MAGMA_COMMON_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#if defined( _WIN32 ) || defined( _WIN64 )

#  include "magmawinthread.h"
#  include <windows.h>
#  include <limits.h>
#  include <io.h>

#else

#  include <pthread.h>
#  include <unistd.h>
#  include <inttypes.h>

#endif

#include "magma.h"
#include "magma_lapack.h"
#include "operators.h"
#include "transpose.h"

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
 *  Determine if weak symbol are allowed 
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
 *    - sqrt alone cannot be catched by the generation script because of tsqrt
 */

#define magma_dsqrt sqrt
#define magma_ssqrt sqrt

#endif /* _MAGMA_COMMON_H_ */
