/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mathieu Faverge
       @author Mark Gates

       Based on PLASMA common.h
*/

/***************************************************************************//**
 *  MAGMA facilities of interest to both src and magmablas directories
 **/
#ifndef MAGMA_INTERNAL_H
#define MAGMA_INTERNAL_H

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
    // in both magma_internal.h and testings.h
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
    
    // our magma_winthread doesn't have pthread_key;
    // assume other platforms (Linux, MacOS, etc.) do.
    #define HAVE_PTHREAD_KEY

#endif

// provide our own support for pthread_barrier on MacOS and Windows
#include "pthread_barrier.h"

// MAGMA_SOURCE is defined in common_magma.h, so it can #include this file but get the right magma.h header.
#if MAGMA_SOURCE == 1
    #include "magma.h"
#else
    #include "magma_v2.h"
#endif

#include "magma_lapack.h"
#include "magma_operators.h"
#include "transpose.h"
#include "magma_threadsetting.h"

/** ****************************************************************************
 *  Define magma_queue structure
 */

struct magma_queue
{
#ifdef __cplusplus
public:
    /* getters */
    int              device()          { return device__;   }
    cudaStream_t     cuda_stream()     { return stream__;   }
    cublasHandle_t   cublas_handle()   { return cublas__;   }
    cusparseHandle_t cusparse_handle() { return cusparse__; }
    
protected:
    friend
    void magma_queue_create_v2_internal(
        magma_device_t device, magma_queue_t* queuePtr,
        const char* func, const char* file, int line );
    
    friend
    void magma_queue_create_from_cuda_internal(
        magma_device_t   device,
        cudaStream_t     stream,
        cublasHandle_t   cublas_handle,
        cusparseHandle_t cusparse_handle,
        magma_queue_t*   queuePtr,
        const char* func, const char* file, int line );
    
    friend
    void magma_queue_destroy_internal(
        magma_queue_t queue,
        const char* func, const char* file, int line );
#endif // __cplusplus
    
    /* private members -- access through getters */
    int              own__;
    int              device__;
    #ifdef HAVE_CUBLAS
    cudaStream_t     stream__;
    cublasHandle_t   cublas__;
    cusparseHandle_t cusparse__;
    #endif
};

#ifdef __cplusplus
extern "C" {
#endif

// needed for BLAS functions that no longer include magma.h (v1)
magma_queue_t magmablasGetQueue();

#ifdef __cplusplus
}
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
 *  in both magma_internal.h and testings.h
 **/
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// suppress "warning: unused variable" in a portable fashion
#define MAGMA_UNUSED(var)  ((void)var)

#endif /* MAGMA_INTERNAL_H */
