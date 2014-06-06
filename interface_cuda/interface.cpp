/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
 
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>

#include "common_magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// --------------------
// subset of the CUDA device properties, set by magma_init()
struct magma_device
{
    size_t memory;
    magma_int_t cuda_arch;
};

int g_magma_devices_cnt = 0;
struct magma_device* g_magma_devices = NULL;


// ========================================
// initialization
// --------------------
// Caches information about available CUDA devices.
// When renumbering devices after calling magma_init,
// call magma_finalize, then cudaSetValidDevices, then magma_init again.
// Ideally magma_init is paired with magma_finalize, but this implementation
// ensures there isn't a memory leak if magma_init is called multiple times
// without calling magma_finalize.
extern "C"
magma_err_t magma_init()
{
    if ( g_magma_devices == NULL ) {
        cudaGetDeviceCount( &g_magma_devices_cnt );
        g_magma_devices = (struct magma_device*) malloc( g_magma_devices_cnt * sizeof(struct magma_device) );
        for( int i = 0; i < g_magma_devices_cnt; ++i ) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties( &prop, i );
            g_magma_devices[i].memory = prop.totalGlobalMem;
            g_magma_devices[i].cuda_arch  = prop.major*100 + prop.minor*10;
        }
    }
    return MAGMA_SUCCESS;
}

// --------------------
// Frees information about CUDA devices.
extern "C"
magma_err_t magma_finalize()
{
    free( g_magma_devices );
    g_magma_devices = NULL;
    return MAGMA_SUCCESS;
}

// --------------------
// Print the available GPU devices. Used in testing.
extern "C"
void magma_print_devices()
{
    int major, minor, micro;
    magma_version( &major, &minor, &micro );
    printf( "MAGMA %d.%d.%d %s, compiled for CUDA capability >= %.1f\n",
            major, minor, micro, MAGMA_VERSION_STAGE, MIN_CUDA_ARCH/100. );
    
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    for( int idevice = 0; idevice < ndevices; idevice++ ) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, idevice );
        printf( "device %d: %s, %.1f MHz clock, %.1f MB memory, capability %d.%d\n",
                idevice,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.major,
                prop.minor );
        
        if ( prop.major*100 + prop.minor*10 < MIN_CUDA_ARCH ) {
            printf("\n"
                   "==============================================================================\n"
                   "WARNING: MAGMA was compiled only for higher CUDA capability %.1f and higher;\n"
                   "some routines will not run correctly!\n"
                   "==============================================================================\n\n",
                   MIN_CUDA_ARCH/100. );
        }
    }
}


// ========================================
// device support
// ---------------------------------------------
// Returns CUDA architecture capability for the current device.
// This requires magma_init to be called first (to cache the information).
// Version is integer xyz, where x is major, y is minor, and z is micro,
// the same as __CUDA_ARCH__. For instance, for architecture 1.3, returns 130.
extern "C"
magma_int_t magma_getdevice_arch()
{
    int dev;
    cudaGetDevice( &dev );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].cuda_arch;
}


// --------------------
extern "C"
void magma_getdevices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    numPtr )
{
    cudaError_t err;
    int cnt;
    err = cudaGetDeviceCount( &cnt );
    check_error( err );
    
    cnt = min( cnt, size );
    for( int i = 0; i < cnt; ++i ) {
        devices[i] = i;
    }
    *numPtr = cnt;
}

// --------------------
extern "C"
void magma_getdevice( magma_device_t* device )
{
    cudaError_t err;
    err = cudaGetDevice( device );
    check_error( err );
}

// --------------------
extern "C"
void magma_setdevice( magma_device_t device )
{
    cudaError_t err;
    err = cudaSetDevice( device );
    check_error( err );
}

// --------------------
extern "C"
void magma_device_sync()
{
    cudaError_t err;
    err = cudaDeviceSynchronize();
    check_error( err );
}


// ========================================
// queue support
// At the moment, MAGMA queue == CUDA stream.
// In the future, MAGMA queue may be CUBLAS handle.
// --------------------
extern "C"
void magma_queue_create_internal(
    /*magma_device_t device,*/ magma_queue_t* queuePtr,
    const char* func, const char* file, int line )    
{
    //cudaStream_t   stream;
    //cublasStatus_t stat;
    cudaError_t    err;
    //err  = cudaSetDevice( device );
    //stat = cublasCreate( queuePtr );
    err  = cudaStreamCreate( queuePtr );  //&stream );
    //stat = cublasSetStream( *queuePtr, stream );
    check_xerror( err, func, file, line );
}

// --------------------
extern "C"
void magma_queue_destroy_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    //cudaStream_t   stream;
    //cublasStatus_t stat;
    cudaError_t    err;
    //stat = cublasGetStream( queue, &stream );
    err  = cudaStreamDestroy( queue );  //stream );
    //stat = cublasDestroy( queue );
    check_xerror( err, func, file, line );
}

// --------------------
extern "C"
void magma_queue_sync_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    //cudaStream_t   stream;
    //cublasStatus_t stat;
    cudaError_t    err;
    //stat = cublasGetStream( queue, &stream );
    err  = cudaStreamSynchronize( queue );  //stream );
    check_xerror( err, func, file, line );
}


// ========================================
// event support
// --------------------
extern "C"
void magma_event_create( magma_event_t* event )
{
    cudaError_t err;
    err = cudaEventCreate( event );
    check_error( err );
}

// --------------------
extern "C"
void magma_event_destroy( magma_event_t event )
{
    cudaError_t err;
    err = cudaEventDestroy( event );
    check_error( err );
}

// --------------------
extern "C"
void magma_event_record( magma_event_t event, magma_queue_t queue )
{
    cudaError_t err;
    err = cudaEventRecord( event, queue );
    check_error( err );
}

// --------------------
// blocks CPU until event occurs
extern "C"
void magma_event_sync( magma_event_t event )
{
    cudaError_t err;
    err = cudaEventSynchronize( event );
    check_error( err );
}

// --------------------
// blocks queue (but not CPU) until event occurs
extern "C"
void magma_queue_wait_event( magma_queue_t queue, magma_event_t event )
{
    cudaError_t err;
    err = cudaStreamWaitEvent( queue, event, 0 );
    check_error( err );
}

#endif // HAVE_CUBLAS
