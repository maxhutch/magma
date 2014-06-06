#ifndef TESTINGS_H
#define TESTINGS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "magma.h"


/***************************************************************************//**
 *  For portability to Windows
 */
#if defined( _WIN32 ) || defined( _WIN64 )
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

#ifndef roundup
#define roundup(a, b) (b <= 0) ? (a) : (((a) + (b)-1) & ~((b)-1))
#endif

#ifndef ceildiv
#define ceildiv(a, b) ((a - 1)/b + 1)
#endif


/***************************************************************************//**
 * Macros to handle error checking.
 */

#define TESTING_INIT()                                                     \
    magma_init();                                                          \
    if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {                          \
        fprintf(stderr, "ERROR: cublasInit failed\n");                     \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }                                                                      \
    magma_print_devices();


#define TESTING_FINALIZE()                                                 \
    magma_finalize();                                                      \
    cublasShutdown();


#define TESTING_INIT_MGPU()                                                \
{                                                                          \
    magma_init();                                                          \
    int ndevices;                                                          \
    cudaGetDeviceCount( &ndevices );                                       \
    for( int idevice = 0; idevice < ndevices; ++idevice ) {                \
        magma_setdevice( idevice );                                        \
        if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {                      \
            fprintf(stderr, "ERROR: gpu %d: cublasInit failed\n", idevice);\
            magma_finalize();                                              \
            exit(-1);                                                      \
        }                                                                  \
    }                                                                      \
    magma_setdevice(0);                                                    \
    magma_print_devices();                                                 \
}


#define TESTING_FINALIZE_MGPU()                                            \
{                                                                          \
    magma_finalize();                                                      \
    int ndevices;                                                          \
    cudaGetDeviceCount( &ndevices );                                       \
    for( int idevice = 0; idevice < ndevices; ++idevice ) {                \
        magma_setdevice(idevice);                                          \
        cublasShutdown();                                                  \
    }                                                                      \
}


#define TESTING_MALLOC_CPU( ptr, type, size )                              \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc_cpu( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr ); \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }


#define TESTING_MALLOC_PIN( ptr, type, size )                                 \
    if ( MAGMA_SUCCESS !=                                                     \
            magma_malloc_pinned( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! magma_malloc_pinned failed for: %s\n", #ptr ); \
        magma_finalize();                                                     \
        exit(-1);                                                             \
    }


#define TESTING_MALLOC_DEV( ptr, type, size )                              \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc( (void**) &ptr, (size)*sizeof(type) )) {          \
        fprintf( stderr, "!!!! magma_malloc failed for: %s\n", #ptr );     \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }


#define TESTING_FREE_CPU( ptr )                                            \
    magma_free_cpu( ptr )


#define TESTING_FREE_PIN( ptr )                                         \
    magma_free_pinned( ptr )


#define TESTING_FREE_DEV( ptr )                                            \
    magma_free( ptr )


#ifdef __cplusplus
extern "C" {
#endif

void magma_zmake_hermitian( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_cmake_hermitian( magma_int_t N, magmaFloatComplex*  A, magma_int_t lda );
void magma_dmake_symmetric( magma_int_t N, double*             A, magma_int_t lda );
void magma_smake_symmetric( magma_int_t N, float*              A, magma_int_t lda );

void magma_zmake_hpd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_cmake_hpd( magma_int_t N, magmaFloatComplex*  A, magma_int_t lda );
void magma_dmake_hpd( magma_int_t N, double*             A, magma_int_t lda );
void magma_smake_hpd( magma_int_t N, float*              A, magma_int_t lda );

void magma_assert( bool condition, const char* msg, ... );

#define MAX_NTEST 1000

typedef struct magma_opts
{
    // matrix size
    magma_int_t ntest;
    magma_int_t msize[ MAX_NTEST ];
    magma_int_t nsize[ MAX_NTEST ];
    magma_int_t ksize[ MAX_NTEST ];
    magma_int_t mmax;
    magma_int_t nmax;
    magma_int_t kmax;
    
    // scalars
    magma_int_t device;
    magma_int_t pad;
    magma_int_t nb;
    magma_int_t nrhs;
    magma_int_t nstream;
    magma_int_t ngpu;
    magma_int_t niter;
    magma_int_t nthread;
    magma_int_t itype;     // hegvd: problem type
    magma_int_t svd_work;  // gesvd
    magma_int_t version;   // hemm_mgpu, hetrd
    double      fraction;  // hegvdx
    double      tolerance;
    magma_int_t panel_nthread; //first dimension for a 2D big panel
    
    // boolean arguments
    int check;
    int lapack;
    int warmup;
    int all;
    
    // lapack flags
    magma_uplo_t    uplo;
    magma_trans_t   transA;
    magma_trans_t   transB;
    magma_side_t    side;
    magma_diag_t    diag;
    magma_vec_t     jobu;    // gesvd:  no left  singular vectors
    magma_vec_t     jobvt;   // gesvd:  no right singular vectors
    magma_vec_t     jobz;    // heev:   no eigen vectors
    magma_vec_t     jobvr;   // geev:   no right eigen vectors
    magma_vec_t     jobvl;   // geev:   no left  eigen vectors
} magma_opts;

void parse_opts( int argc, char** argv, magma_opts *opts );

#ifdef __cplusplus
}
#endif

#endif /* TESTINGS_H */
