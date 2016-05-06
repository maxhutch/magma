/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
 
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <map>

#ifdef DEBUG_MEMORY
#include <mutex>  // requires C++11
#endif

#include "magma_v2.h"
#include "error.h"

#ifdef HAVE_CUBLAS


#ifdef DEBUG_MEMORY
std::mutex                g_pointers_mutex;  // requires C++11
std::map< void*, size_t > g_pointers_dev;
std::map< void*, size_t > g_pointers_cpu;
std::map< void*, size_t > g_pointers_pin;
#endif


// ========================================
// memory allocation
// Allocate size bytes on GPU, returning pointer in ptrPtr.
extern "C" magma_int_t
magma_malloc( magma_ptr* ptrPtr, size_t size )
{
    // CUDA can't allocate 0 bytes, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
    if ( cudaSuccess != cudaMalloc( ptrPtr, size )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }
    
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_dev[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif
    
    return MAGMA_SUCCESS;
}

// --------------------
// Free GPU memory allocated by magma_malloc.
extern "C" magma_int_t
magma_free_internal( magma_ptr ptr,
    const char* func, const char* file, int line )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_dev.count( ptr ) == 0 ) {
        fprintf( stderr, "magma_free( %p ) that wasn't allocated with magma_malloc.\n", ptr );
    }
    else {
        g_pointers_dev.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif
    
    cudaError_t err = cudaFree( ptr );
    check_xerror( err, func, file, line );
    if ( err != cudaSuccess ) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}

// --------------------
// Allocate size bytes on CPU, returning pointer in ptrPtr.
// The purpose of using this instead of malloc is to properly align arrays
// for vector (SSE) instructions. The default implementation uses
// posix_memalign (on Linux, MacOS, etc.) or _aligned_malloc (on Windows)
// to align memory to a 32 byte boundary.
// Use magma_free_cpu() to free this memory.
extern "C" magma_int_t
magma_malloc_cpu( void** ptrPtr, size_t size )
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
#if 1
#if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = _aligned_malloc( size, 32 );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#else
    int err = posix_memalign( ptrPtr, 32, size );
    if ( err != 0 ) {
        *ptrPtr = NULL;
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
#else
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
    
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_cpu[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif
    
    return MAGMA_SUCCESS;
}

// --------------------
// Free CPU pinned memory previously allocated by magma_malloc_pinned.
// The default implementation uses free(), which works for both malloc and posix_memalign.
// For Windows, _aligned_free() is used.
extern "C" magma_int_t
magma_free_cpu( void* ptr )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_cpu.count( ptr ) == 0 ) {
        fprintf( stderr, "magma_free_cpu( %p ) that wasn't allocated with magma_malloc_cpu.\n", ptr );
    }
    else {
        g_pointers_cpu.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif
    
#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free( ptr );
#endif
    return MAGMA_SUCCESS;
}

// --------------------
// Allocate size bytes on CPU in pinned memory, returning pointer in ptrPtr.
extern "C" magma_int_t
magma_malloc_pinned( void** ptrPtr, size_t size )
{
    // CUDA can't allocate 0 bytes, so allocate some minimal size
    // (for pinned memory, the error is detected in free)
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
    if ( cudaSuccess != cudaMallocHost( ptrPtr, size )) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_pin[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif
    
    return MAGMA_SUCCESS;
}

// --------------------
// Free CPU pinned memory previously allocated by magma_malloc_pinned.
extern "C" magma_int_t
magma_free_pinned_internal( void* ptr,
    const char* func, const char* file, int line )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_pin.count( ptr ) == 0 ) {
        fprintf( stderr, "magma_free_pinned( %p ) that wasn't allocated with magma_malloc_pinned.\n", ptr );
    }
    else {
        g_pointers_pin.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif
    
    cudaError_t err = cudaFreeHost( ptr );
    check_xerror( err, func, file, line );
    if ( cudaSuccess != err ) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}

#endif // HAVE_CUBLAS
