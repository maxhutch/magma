#ifndef TESTINGS_H
#define TESTINGS_H

#include <stdio.h>
#include <stdlib.h>

#include "magma_v2.h"


/***************************************************************************//**
 *  For portability to Windows
 */
#if defined( _WIN32 ) || defined( _WIN64 )
    // functions where Microsoft fails to provide C99 or POSIX standard
    // (only with Microsoft, not with nvcc on Windows)
    // in both magma_internal.h and testings.h
    #ifndef __NVCC__
    
        #include <float.h>
        #define copysign(x,y) _copysign(x,y)
        #define isnan(x)      _isnan(x)
        #define isinf(x)      ( ! _finite(x) && ! _isnan(x) )
        #define isfinite(x)   _finite(x)
        // note _snprintf has slightly different semantics than snprintf
        #define snprintf      _snprintf
        #define unlink        _unlink
        
    #endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

void flops_init();

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


/***************************************************************************//**
 * Macros to handle error checking.
 */

#define TESTING_INIT()                                                     \
    magma_init();                                                          \
    flops_init();                                                          \
    magma_print_environment();

#define TESTING_FINALIZE()                                                 \
    magma_finalize();


/******************* CPU memory */
#define TESTING_MALLOC_CPU( ptr, type, size )                              \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc_cpu( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr ); \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }

#define TESTING_FREE_CPU( ptr ) magma_free_cpu( ptr )


/******************* Pinned CPU memory */
#ifdef HAVE_CUBLAS
    // In CUDA, this allocates pinned memory.
    #define TESTING_MALLOC_PIN( ptr, type, size )                                 \
        if ( MAGMA_SUCCESS !=                                                     \
                magma_malloc_pinned( (void**) &ptr, (size)*sizeof(type) )) {      \
            fprintf( stderr, "!!!! magma_malloc_pinned failed for: %s\n", #ptr ); \
            magma_finalize();                                                     \
            exit(-1);                                                             \
        }
    
    #define TESTING_FREE_PIN( ptr ) magma_free_pinned( ptr )
#else
    // For OpenCL, we don't support pinned memory yet.
    #define TESTING_MALLOC_PIN( ptr, type, size )                              \
        if ( MAGMA_SUCCESS !=                                                  \
                magma_malloc_cpu( (void**) &ptr, (size)*sizeof(type) )) {      \
            fprintf( stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr ); \
            magma_finalize();                                                  \
            exit(-1);                                                          \
        }
    
    #define TESTING_FREE_PIN( ptr ) magma_free_cpu( ptr )
#endif


/******************* GPU memory */
#ifdef HAVE_CUBLAS
    // In CUDA, this has (void**) cast.
    #define TESTING_MALLOC_DEV( ptr, type, size )                              \
        if ( MAGMA_SUCCESS !=                                                  \
                magma_malloc( (void**) &ptr, (size)*sizeof(type) )) {          \
            fprintf( stderr, "!!!! magma_malloc failed for: %s\n", #ptr );     \
            magma_finalize();                                                  \
            exit(-1);                                                          \
        }
#else
    // For OpenCL, ptr is cl_mem* and there is no cast.
    #define TESTING_MALLOC_DEV( ptr, type, size )                              \
        if ( MAGMA_SUCCESS !=                                                  \
                magma_malloc( &ptr, (size)*sizeof(type) )) {                   \
            fprintf( stderr, "!!!! magma_malloc failed for: %s\n", #ptr );     \
            magma_finalize();                                                  \
            exit(-1);                                                          \
        }
#endif

#define TESTING_FREE_DEV( ptr ) magma_free( ptr )


/***************************************************************************//**
 * Functions and data structures used for testing.
 */
void magma_zmake_symmetric( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_cmake_symmetric( magma_int_t N, magmaFloatComplex*  A, magma_int_t lda );
void magma_zmake_hermitian( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_cmake_hermitian( magma_int_t N, magmaFloatComplex*  A, magma_int_t lda );
void magma_dmake_symmetric( magma_int_t N, double*             A, magma_int_t lda );
void magma_smake_symmetric( magma_int_t N, float*              A, magma_int_t lda );

void magma_zmake_spd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_cmake_spd( magma_int_t N, magmaFloatComplex*  A, magma_int_t lda );
void magma_zmake_hpd( magma_int_t N, magmaDoubleComplex* A, magma_int_t lda );
void magma_cmake_hpd( magma_int_t N, magmaFloatComplex*  A, magma_int_t lda );
void magma_dmake_hpd( magma_int_t N, double*             A, magma_int_t lda );
void magma_smake_hpd( magma_int_t N, float*              A, magma_int_t lda );

void magma_assert( bool condition, const char* msg, ... );

void magma_assert_warn( bool condition, const char* msg, ... );

// work around MKL bug in multi-threaded lanhe/lansy
double safe_lapackf77_zlanhe(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const magmaDoubleComplex *A, const magma_int_t *lda,
    double *work );

float  safe_lapackf77_clanhe(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const magmaFloatComplex *A, const magma_int_t *lda,
    float *work );

double safe_lapackf77_dlansy(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const double *A, const magma_int_t *lda,
    double *work );

float safe_lapackf77_slansy(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const float *A, const magma_int_t *lda,
    float *work );

#define MAX_NTEST 1050

typedef enum {
    MagmaOptsDefault = 0,
    MagmaOptsBatched = 1000
} magma_opts_t;

class magma_opts
{
public:
    // constructor
    magma_opts( magma_opts_t flag=MagmaOptsDefault );
    
    // parse command line
    void parse_opts( int argc, char** argv );
    
    // deallocate queues, etc.
    void cleanup();
    
    // matrix size
    magma_int_t ntest;
    magma_int_t msize[ MAX_NTEST ];
    magma_int_t nsize[ MAX_NTEST ];
    magma_int_t ksize[ MAX_NTEST ];
    magma_int_t batchcount;
    
    magma_int_t default_nstart;
    magma_int_t default_nend;
    magma_int_t default_nstep;
    
    // scalars
    magma_int_t device;
    magma_int_t align;
    magma_int_t nb;
    magma_int_t nrhs;
    magma_int_t nqueue;
    magma_int_t ngpu;
    magma_int_t nsub;
    magma_int_t niter;
    magma_int_t nthread;
    magma_int_t offset;
    magma_int_t itype;     // hegvd: problem type
    magma_int_t svd_work;  // gesvd
    magma_int_t version;   // hemm_mgpu, hetrd
    magma_int_t check;
    double      fraction;  // hegvdx
    double      tolerance;
    
    // boolean arguments
    bool lapack;
    bool warmup;
    bool all;
    bool verbose;
    
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
    
    // queue for default device
    magma_queue_t   queue;
    magma_queue_t   queues2[3];  // 2 queues + 1 extra NULL entry to catch errors
    
    #ifdef HAVE_CUBLAS
    // handle for directly calling cublas
    cublasHandle_t  handle;
    #endif
    
    // misc
    int flock_op;   // shared or exclusive lock
    int flock_fd;   // lock file
};

extern const char* g_platform_str;

#ifdef __cplusplus
}
#endif

#endif /* TESTINGS_H */
