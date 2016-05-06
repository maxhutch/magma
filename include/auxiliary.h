/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
*/

#ifndef MAGMA_AUXILIARY_H
#define MAGMA_AUXILIARY_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


// ========================================
// initialization

magma_int_t magma_init( void );
magma_int_t magma_finalize( void );


// ========================================
// version information

void magma_version( magma_int_t* major, magma_int_t* minor, magma_int_t* micro );
void magma_print_environment();


// ========================================
// timing

real_Double_t magma_wtime( void );
real_Double_t magma_sync_wtime( magma_queue_t queue );


// ========================================
// misc. functions

// magma GPU-complex PCIe connection
magma_int_t magma_buildconnection_mgpu(
    magma_int_t gnode[MagmaMaxGPUs+2][MagmaMaxGPUs+2],
    magma_int_t *ncmplx,
    magma_int_t ngpu );

void magma_indices_1D_bcyclic(
    magma_int_t nb, magma_int_t ngpu, magma_int_t dev,
    magma_int_t j0, magma_int_t j1,
    magma_int_t* dj0, magma_int_t* dj1 );

void magma_swp2pswp(
    magma_trans_t trans, magma_int_t n,
    magma_int_t *ipiv,
    magma_int_t *newipiv );


// ========================================
// get NB blocksize

magma_int_t magma_get_smlsize_divideconquer();


// ========================================
// memory allocation

magma_int_t
magma_malloc( magma_ptr *ptr_ptr, size_t bytes );

magma_int_t
magma_malloc_cpu( void **ptr_ptr, size_t bytes );

magma_int_t
magma_malloc_pinned( void **ptr_ptr, size_t bytes );

magma_int_t
magma_free_cpu( void *ptr );

#define magma_free( ptr ) \
        magma_free_internal( ptr, __func__, __FILE__, __LINE__ )

#define magma_free_pinned( ptr ) \
        magma_free_pinned_internal( ptr, __func__, __FILE__, __LINE__ )

magma_int_t
magma_free_internal(
    magma_ptr ptr,
    const char* func, const char* file, int line );

magma_int_t
magma_free_pinned_internal(
    void *ptr,
    const char* func, const char* file, int line );


// type-safe convenience functions to avoid using (void**) cast and sizeof(...)
// here n is the number of elements (floats, doubles, etc.) not the number of bytes.
static inline magma_int_t magma_imalloc( magmaInt_ptr           *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magma_int_t)        ); }
static inline magma_int_t magma_index_malloc( magmaIndex_ptr    *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magma_index_t)      ); }
static inline magma_int_t magma_smalloc( magmaFloat_ptr         *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(float)              ); }
static inline magma_int_t magma_dmalloc( magmaDouble_ptr        *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(double)             ); }
static inline magma_int_t magma_cmalloc( magmaFloatComplex_ptr  *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_int_t magma_zmalloc( magmaDoubleComplex_ptr *ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(magmaDoubleComplex) ); }

static inline magma_int_t magma_imalloc_cpu( magma_int_t        **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magma_int_t)        ); }
static inline magma_int_t magma_index_malloc_cpu( magma_index_t **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magma_index_t)      ); }
static inline magma_int_t magma_smalloc_cpu( float              **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(float)              ); }
static inline magma_int_t magma_dmalloc_cpu( double             **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(double)             ); }
static inline magma_int_t magma_cmalloc_cpu( magmaFloatComplex  **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_int_t magma_zmalloc_cpu( magmaDoubleComplex **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(magmaDoubleComplex) ); }

static inline magma_int_t magma_imalloc_pinned( magma_int_t        **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magma_int_t)        ); }
static inline magma_int_t magma_index_malloc_pinned( magma_index_t **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magma_index_t)      ); }
static inline magma_int_t magma_smalloc_pinned( float              **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(float)              ); }
static inline magma_int_t magma_dmalloc_pinned( double             **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(double)             ); }
static inline magma_int_t magma_cmalloc_pinned( magmaFloatComplex  **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_int_t magma_zmalloc_pinned( magmaDoubleComplex **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(magmaDoubleComplex) ); }

magma_int_t magma_is_devptr( const void* ptr );


// ========================================
// device support

magma_int_t magma_num_gpus( void );
/* todo: num_accelerators */
/* todo: total accelerators? available accelerators? i.e., number to use vs. number available. */

magma_int_t magma_getdevice_arch();
/* magma_int_t magma_getdevice_arch( magma_int_t dev );   todo: new */

void magma_getdevices(
    magma_device_t* devices,
    magma_int_t  size,
    magma_int_t* num_ptr );

void magma_getdevice( magma_device_t* dev );

void magma_setdevice( magma_device_t dev );


// ========================================
// event support

void magma_event_create( magma_event_t* event_ptr );

void magma_event_destroy( magma_event_t event );

void magma_event_record( magma_event_t event, magma_queue_t queue );

void magma_event_query( magma_event_t event );

// blocks CPU until event occurs
void magma_event_sync( magma_event_t event );

// blocks queue (but not CPU) until event occurs
void magma_queue_wait_event( magma_queue_t queue, magma_event_t event );


// ========================================
// error handler

void magma_xerbla( const char *name, magma_int_t info );

const char* magma_strerror( magma_int_t error );


// ========================================
// string functions

size_t magma_strlcpy( char *dst, const char *src, size_t size );


// ========================================
// integer functions

/// For integers x >= 0, y > 0, returns ceil( x/y ).
/// For x == 0, this is 0.
__host__ __device__
static inline magma_int_t magma_ceildiv( magma_int_t x, magma_int_t y )
{
    return (x + y - 1)/y;
}

/// For integers x >= 0, y > 0, returns x rounded up to multiple of y.
/// For x == 0, this is 0.
/// This implementation does not assume y is a power of 2.
__host__ __device__
static inline magma_int_t magma_roundup( magma_int_t x, magma_int_t y )
{
    return magma_ceildiv( x, y ) * y;
}


// ========================================
// scalar functions

// real and complex square root
// sqrt alone cannot be caught by the generation script because of tsqrt
static inline float  magma_ssqrt( float  x ) { return sqrtf( x ); }
static inline double magma_dsqrt( double x ) { return sqrt( x ); }
magmaFloatComplex    magma_csqrt( magmaFloatComplex  x );
magmaDoubleComplex   magma_zsqrt( magmaDoubleComplex x );

double magma_cabs ( magmaDoubleComplex x );
float  magma_cabsf( magmaFloatComplex  x );


#ifdef __cplusplus
}
#endif


#endif  // MAGMA_AUXILIARY_H
