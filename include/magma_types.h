/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
*/

#ifndef MAGMA_TYPES_H
#define MAGMA_TYPES_H

#include <stdint.h>
#include <assert.h>


// ========================================
// C99 standard defines __func__. Some older compilers use __FUNCTION__.
// Note __func__ in C99 is not a macro, so ifndef __func__ doesn't work.
#if __STDC_VERSION__ < 199901L
  #ifndef __func__
    #if __GNUC__ >= 2 || _MSC_VER >= 1300
      #define __func__ __FUNCTION__
    #else
      #define __func__ "<unknown>"
    #endif
  #endif
#endif


// ========================================
// To use int64_t, link with mkl_intel_ilp64 or similar (instead of mkl_intel_lp64).
#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
typedef int64_t magma_int_t;
typedef int64_t magma_err_t;
#else
typedef int magma_int_t;
typedef int magma_err_t;
#endif

// Define new type that the precision generator will not change (matches PLASMA)
typedef double real_Double_t;


// ========================================
// define types specific to implementation (CUDA, OpenCL, MIC)
// define macros to deal with complex numbers
#if HAVE_CUBLAS
    #ifndef CUBLAS_V2_H_
    #include <cublas.h>
    #endif
    
    typedef cudaStream_t   magma_queue_t;
    typedef cudaEvent_t    magma_event_t;
    typedef int            magma_device_t;
    
    typedef cuDoubleComplex magmaDoubleComplex;
    typedef cuFloatComplex  magmaFloatComplex;
    
    #define MAGMA_Z_MAKE(r,i)     make_cuDoubleComplex(r, i)
    #define MAGMA_Z_REAL(a)       (a).x
    #define MAGMA_Z_IMAG(a)       (a).y
    #define MAGMA_Z_SET2REAL(a,r) { (a).x = (r);   (a).y = 0.0; }
    #define MAGMA_Z_ADD(a, b)     cuCadd(a, b)
    #define MAGMA_Z_SUB(a, b)     cuCsub(a, b)
    #define MAGMA_Z_MUL(a, b)     cuCmul(a, b)
    #define MAGMA_Z_DIV(a, b)     cuCdiv(a, b)
    #define MAGMA_Z_ABS(a)        cuCabs(a)
    #define MAGMA_Z_CNJG(a)       cuConj(a)
    #define MAGMA_Z_DSCALE(v,t,s) {(v).x = (t).x/(s); (v).y = (t).y/(s);}
    
    #define MAGMA_C_MAKE(r,i)     make_cuFloatComplex(r, i)
    #define MAGMA_C_REAL(a)       (a).x
    #define MAGMA_C_IMAG(a)       (a).y
    #define MAGMA_C_SET2REAL(a,r) { (a).x = (r);   (a).y = 0.0; }
    #define MAGMA_C_ADD(a, b)     cuCaddf(a, b)
    #define MAGMA_C_SUB(a, b)     cuCsubf(a, b)
    #define MAGMA_C_MUL(a, b)     cuCmulf(a, b)
    #define MAGMA_C_DIV(a, b)     cuCdivf(a, b)
    #define MAGMA_C_ABS(a)        cuCabsf(a)
    #define MAGMA_C_CNJG(a)       cuConjf(a)
    #define MAGMA_C_SSCALE(v,t,s) {(v).x = (t).x/(s); (v).y = (t).y/(s);}
    
#elif HAVE_clAmdBlas
    #if defined(__APPLE__) || defined(__MACOSX)
    #include "my_amdblas.h"
    #else
    #include <clAmdBlas.h>
    #endif
    
    typedef cl_command_queue  magma_queue_t;
    typedef cl_event          magma_event_t;
    typedef cl_device_id      magma_device_t;
    
    typedef DoubleComplex magmaDoubleComplex;
    typedef FloatComplex  magmaFloatComplex;
    
    #define MAGMA_Z_MAKE(r,i)     doubleComplex(r,i)
    #define MAGMA_Z_REAL(a)       (a).x
    #define MAGMA_Z_IMAG(a)       (a).y
    #define MAGMA_Z_SET2REAL(a,r) { (a).x = (r);   (a).y = 0.0; }
    #define MAGMA_Z_ADD(a, b)     MAGMA_Z_MAKE((a).x+(b).x, (a).y+(b).y)
    #define MAGMA_Z_SUB(a, b)     MAGMA_Z_MAKE((a).x-(b).x, (a).y-(b).y)
    #define MAGMA_Z_ABS(a)        magma_cabs(a)
    #define MAGMA_Z_CNJG(a)       MAGMA_Z_MAKE((a).x, -(a).y)
    #define MAGMA_Z_DSCALE(v,t,s) {(v).x = (t).x/(s); (v).y = (t).y/(s);}
    
    #define MAGMA_C_MAKE(r,i)     floatComplex(r,i)
    #define MAGMA_C_REAL(a)       (a).x
    #define MAGMA_C_IMAG(a)       (a).y
    #define MAGMA_C_SET2REAL(a,r) { (a).x = (r);   (a).y = 0.0; }
    #define MAGMA_C_ADD(a, b)     MAGMA_C_MAKE((a).x+(b).x, (a).y+(b).y)
    #define MAGMA_C_SUB(a, b)     MAGMA_C_MAKE((a).x-(b).x, (a).y-(b).y)
    #define MAGMA_C_ABS(a)        magma_cabsf(a)
    #define MAGMA_C_CNJG(a)       MAGMA_C_MAKE((a).x, -(a).y)
    #define MAGMA_C_SSCALE(v,t,s) {(v).x = (t).x/(s); (v).y = (t).y/(s);}

#elif HAVE_MIC
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <string.h>
    #include <sys/mman.h>
    #include <sys/ioctl.h>
    #include <sys/time.h>
    #include <scif.h>
    //#include <mkl.h>

    typedef int   magma_queue_t;
    typedef int   magma_event_t;
    typedef int   magma_device_t;

    #include <complex>
    typedef std::complex<float>   magmaFloatComplex;
    typedef std::complex<double>  magmaDoubleComplex;

    #define MAGMA_Z_MAKE(r, i)    std::complex<double>(r,i)
    #define MAGMA_Z_REAL(x)       (x).real()
    #define MAGMA_Z_IMAG(x)       (x).imag()
    #define MAGMA_Z_SET2REAL(a,r) { (a).real() = (r);   (a).imag() = 0.0; }
    #define MAGMA_Z_ADD(a, b)     ((a)+(b))
    #define MAGMA_Z_SUB(a, b)     ((a)-(b))
    #define MAGMA_Z_MUL(a, b)     ((a)*(b))
    #define MAGMA_Z_DIV(a, b)     ((a)/(b))
    #define MAGMA_Z_CNJG(a)       conj(a)
    #define MAGMA_Z_DSCALE(v,t,s) ((v) = (t)/(s))

    #define MAGMA_C_MAKE(r, i)    std::complex<float> (r,i)
    #define MAGMA_C_REAL(x)       (x).real()
    #define MAGMA_C_IMAG(x)       (x).imag()
    #define MAGMA_C_SET2REAL(a,r) { (a).real() = (r);   (a).imag() = 0.0; }
    #define MAGMA_C_ADD(a, b)     ((a)+(b))
    #define MAGMA_C_SUB(a, b)     ((a)-(b))
    #define MAGMA_C_MUL(a, b)     ((a)*(b))
    #define MAGMA_C_DIV(a, b)     ((a)/(b))
    #define MAGMA_C_CNJG(a)       conj(a)
    #define MAGMA_C_SSCALE(v,t,s) ((v) = (t)/(s))
#else
    #error "One of HAVE_CUBLAS, HAVE_clAmdBlas, or HAVE_MIC must be defined. This typically happens in Makefile.internal."
#endif

#define MAGMA_Z_EQUAL(a,b)        (MAGMA_Z_REAL(a)==MAGMA_Z_REAL(b) && MAGMA_Z_IMAG(a)==MAGMA_Z_IMAG(b))
#define MAGMA_Z_NEGATE(a)         MAGMA_Z_MAKE( -MAGMA_Z_REAL(a), -MAGMA_Z_IMAG(a))

#define MAGMA_C_EQUAL(a,b)        (MAGMA_C_REAL(a)==MAGMA_C_REAL(b) && MAGMA_C_IMAG(a)==MAGMA_C_IMAG(b))
#define MAGMA_C_NEGATE(a)         MAGMA_C_MAKE( -MAGMA_C_REAL(a), -MAGMA_C_IMAG(a))

#define MAGMA_D_MAKE(r,i)         (r)
#define MAGMA_D_REAL(x)           (x)
#define MAGMA_D_IMAG(x)           (0.0)
#define MAGMA_D_SET2REAL(a,r)     (a) = (r)
#define MAGMA_D_ADD(a, b)         ((a) + (b))
#define MAGMA_D_SUB(a, b)         ((a) - (b))
#define MAGMA_D_MUL(a, b)         ((a) * (b))
#define MAGMA_D_DIV(a, b)         ((a) / (b))
#define MAGMA_D_ABS(a)            ((a)>0?(a):-(a))
#define MAGMA_D_CNJG(a)           (a)
#define MAGMA_D_EQUAL(a,b)        ((a) == (b))
#define MAGMA_D_NEGATE(a)         (-a)
#define MAGMA_D_DSCALE(v, t, s)   (v) = (t)/(s)

#define MAGMA_S_MAKE(r,i)         (r)
#define MAGMA_S_REAL(x)           (x)
#define MAGMA_S_IMAG(x)           (0.0)
#define MAGMA_S_SET2REAL(a,r)     (a) = (r)
#define MAGMA_S_ADD(a, b)         ((a) + (b))
#define MAGMA_S_SUB(a, b)         ((a) - (b))
#define MAGMA_S_MUL(a, b)         ((a) * (b))
#define MAGMA_S_DIV(a, b)         ((a) / (b))
#define MAGMA_S_ABS(a)            ((a)>0?(a):-(a))
#define MAGMA_S_CNJG(a)           (a)
#define MAGMA_S_EQUAL(a,b)        ((a) == (b))
#define MAGMA_S_NEGATE(a)         (-a)
#define MAGMA_S_SSCALE(v, t, s)   (v) = (t)/(s)

#define MAGMA_Z_ZERO              MAGMA_Z_MAKE( 0.0, 0.0)
#define MAGMA_Z_ONE               MAGMA_Z_MAKE( 1.0, 0.0)
#define MAGMA_Z_HALF              MAGMA_Z_MAKE( 0.5, 0.0)
#define MAGMA_Z_NEG_ONE           MAGMA_Z_MAKE(-1.0, 0.0)
#define MAGMA_Z_NEG_HALF          MAGMA_Z_MAKE(-0.5, 0.0)

#define MAGMA_C_ZERO              MAGMA_C_MAKE( 0.0, 0.0)
#define MAGMA_C_ONE               MAGMA_C_MAKE( 1.0, 0.0)
#define MAGMA_C_HALF              MAGMA_C_MAKE( 0.5, 0.0)
#define MAGMA_C_NEG_ONE           MAGMA_C_MAKE(-1.0, 0.0)
#define MAGMA_C_NEG_HALF          MAGMA_C_MAKE(-0.5, 0.0)

#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_ONE               ( 1.0)
#define MAGMA_D_HALF              ( 0.5)
#define MAGMA_D_NEG_ONE           (-1.0)
#define MAGMA_D_NEG_HALF          (-0.5)

#define MAGMA_S_ZERO              ( 0.0)
#define MAGMA_S_ONE               ( 1.0)
#define MAGMA_S_HALF              ( 0.5)
#define MAGMA_S_NEG_ONE           (-1.0)
#define MAGMA_S_NEG_HALF          (-0.5)

#ifndef CBLAS_SADDR
#define CBLAS_SADDR(a)  &(a)
#endif

#if HAVE_clAmdBlas
    // OpenCL uses opaque memory references on GPU
    typedef cl_mem magma_ptr;
    typedef cl_mem magmaInt_ptr;
    typedef cl_mem magmaFloat_ptr;
    typedef cl_mem magmaDouble_ptr;
    typedef cl_mem magmaFloatComplex_ptr;
    typedef cl_mem magmaDoubleComplex_ptr;
    
    typedef cl_mem magma_const_ptr;
    typedef cl_mem magmaInt_const_ptr;
    typedef cl_mem magmaFloat_const_ptr;
    typedef cl_mem magmaDouble_const_ptr;
    typedef cl_mem magmaFloatComplex_const_ptr;
    typedef cl_mem magmaDoubleComplex_const_ptr;
#else
    // MIC and CUDA use regular pointers on GPU
    typedef void               *magma_ptr;
    typedef magma_int_t        *magmaInt_ptr;
    typedef float              *magmaFloat_ptr;
    typedef double             *magmaDouble_ptr;
    typedef magmaFloatComplex  *magmaFloatComplex_ptr;
    typedef magmaDoubleComplex *magmaDoubleComplex_ptr;
    
    typedef void               const *magma_const_ptr;
    typedef magma_int_t        const *magmaInt_const_ptr;
    typedef float              const *magmaFloat_const_ptr;
    typedef double             const *magmaDouble_const_ptr;
    typedef magmaFloatComplex  const *magmaFloatComplex_const_ptr;
    typedef magmaDoubleComplex const *magmaDoubleComplex_const_ptr;
#endif


// ========================================
// MAGMA constants

// ----------------------------------------
#define MAGMA_VERSION_MAJOR 1
#define MAGMA_VERSION_MINOR 4
#define MAGMA_VERSION_MICRO 1

// stage is "svn", "beta#", "rc#" (release candidate), or blank ("") for final release
#define MAGMA_VERSION_STAGE ""

#define MagmaMaxGPUs 8


// ----------------------------------------
// Return codes
// LAPACK argument errors are < 0 but > MAGMA_ERR.
// MAGMA errors are < MAGMA_ERR.
#define MAGMA_SUCCESS               0
#define MAGMA_ERR                  -100
#define MAGMA_ERR_NOT_INITIALIZED  -101
#define MAGMA_ERR_REINITIALIZED    -102
#define MAGMA_ERR_NOT_SUPPORTED    -103
#define MAGMA_ERR_ILLEGAL_VALUE    -104
#define MAGMA_ERR_NOT_FOUND        -105
#define MAGMA_ERR_ALLOCATION       -106
#define MAGMA_ERR_INTERNAL_LIMIT   -107
#define MAGMA_ERR_UNALLOCATED      -108
#define MAGMA_ERR_FILESYSTEM       -109
#define MAGMA_ERR_UNEXPECTED       -110
#define MAGMA_ERR_SEQUENCE_FLUSHED -111
#define MAGMA_ERR_HOST_ALLOC       -112
#define MAGMA_ERR_DEVICE_ALLOC     -113
#define MAGMA_ERR_CUDASTREAM       -114
#define MAGMA_ERR_INVALID_PTR      -115
#define MAGMA_ERR_UNKNOWN          -116
#define MAGMA_ERR_NOT_IMPLEMENTED  -117


// ----------------------------------------
// parameter constants
// [In the future, these will be numbered, as indicated by comments,
// [instead of character constants.]
// numbering is consistent with CBLAS and PLASMA; see plasma/include/plasma.h
#define MagmaRowMajor      'R'  /* 101 */
#define MagmaColMajor      'C'  /* 102 */

#define MagmaNoTrans       'N'  /* 111 */
#define MagmaTrans         'T'  /* 112 */
#define MagmaConjTrans     'C'  /* 113 */

#define MagmaUpper         'U'  /* 121 */
#define MagmaLower         'L'  /* 122 */
#define MagmaUpperLower    'G'  /* 123 */
#define MagmaFull          'G'  /* 123 */  // see lascl
#define MagmaHessenberg    'H'  /* 124 */  // see lascl

#define MagmaNonUnit       'N'  /* 131 */
#define MagmaUnit          'U'  /* 132 */

#define MagmaLeft          'L'  /* 141 */
#define MagmaRight         'R'  /* 142 */

#define MagmaOneNorm       '1'  /* 171 */
#define MagmaRealOneNorm   172
#define MagmaTwoNorm       '2'  /* 173 */
#define MagmaFrobeniusNorm 'F'  /* 174 */
#define MagmaInfNorm       'I'  /* 175 */
#define MagmaRealInfNorm   176
#define MagmaMaxNorm       'M'  /* 177 */
#define MagmaRealMaxNorm   178

#define MagmaDistUniform   201
#define MagmaDistSymmetric 202
#define MagmaDistNormal    203

#define MagmaHermGeev      241
#define MagmaHermPoev      242
#define MagmaNonsymPosv    243
#define MagmaSymPosv       244

#define MagmaNoPacking     291
#define MagmaPackSubdiag   292
#define MagmaPackSupdiag   293
#define MagmaPackColumn    294
#define MagmaPackRow       295
#define MagmaPackLowerBand 296
#define MagmaPackUpeprBand 297
#define MagmaPackAll       298

#define MagmaNoVec         'N'  /* 301 */  /* geev, syev, gesvd */
#define MagmaVec           'V'  /* 302 */  /* geev, syev */
#define MagmaIVec          'I'  /* 303 */  /* stedc */
#define MagmaAllVec        'A'  /* 304 */  /* gesvd */
#define MagmaSomeVec       'S'  /* 305 */  /* gesvd */
#define MagmaOverwriteVec  'O'  /* 306 */  /* gesvd */

#define MagmaRangeAll      'A'
#define MagmaRangeV        'V'
#define MagmaRangeI        'I'

#define MagmaForward       'F'  /* 391 */  /* larfb */
#define MagmaBackward      'B'  /* 392 */  /* larfb */

#define MagmaColumnwise    'C'  /* 401 */  /* larfb */
#define MagmaRowwise       'R'  /* 402 */  /* larfb */

#define Magma_CSR          411
#define Magma_ELLPACK      412
#define Magma_ELLPACKT     413
#define Magma_DENSE        414  
#define Magma_BCSR         415
#define Magma_CSC          416
#define Magma_HYB          417
#define Magma_COO          418

#define Magma_CPU          421
#define Magma_DEV          422

#define Magma_CG           431
#define Magma_GMRES        432
#define Magma_BICGSTAB     433
#define Magma_JACOBI       434
#define Magma_GS           435

#define Magma_DCOMPLEX     451
#define Magma_FCOMPLEX     452
#define Magma_DOUBLE       453
#define Magma_FLOAT        454


// remember to update min/max when adding constants!
#define MagmaMinConst      101
#define MagmaMaxConst      454


// ----------------------------------------
// these could be enums, but that isn't portable in C++,
// e.g., if -fshort-enums is used
typedef char magma_order_t;
typedef char magma_trans_t;
typedef char magma_uplo_t;
typedef char magma_diag_t;
typedef char magma_side_t;
typedef char magma_type_t;
typedef char magma_norm_t;
typedef char magma_dist_t;
typedef char magma_pack_t;
typedef char magma_vec_t;
typedef char magma_range_t;
typedef char magma_direct_t;
typedef char magma_storev_t;

// for sparse linear algebra
// properties of the magma_sparse_matrix
typedef int magma_storage_t;
typedef int magma_location_t;
// properties of the magma_precond_parameters
typedef int magma_precond_type;
typedef int magma_precision;


// ----------------------------------------
// string constants for calling Fortran BLAS and LAPACK
// todo: use translators instead? lapack_trans_const( MagmaUpper )
#define MagmaRowMajorStr   "Row"
#define MagmaColMajorStr   "Col"

#define MagmaNoTransStr    "NoTrans"
#define MagmaTransStr      "Trans"
#define MagmaConjTransStr  "ConjTrans"

#define MagmaUpperStr      "Upper"
#define MagmaLowerStr      "Lower"
#define MagmaUpperLowerStr "Full"
#define MagmaFullStr       "Full"

#define MagmaNonUnitStr    "NonUnit"
#define MagmaUnitStr       "Unit"

#define MagmaLeftStr       "Left"
#define MagmaRightStr      "Right"

#define MagmaOneNormStr       "1"
#define MagmaTwoNormStr       "2"
#define MagmaFrobeniusNormStr "Fro"
#define MagmaInfNormStr       "Inf"
#define MagmaMaxNormStr       "Max"

#define MagmaForwardStr    "Forward"
#define MagmaBackwardStr   "Backward"

#define MagmaColumnwiseStr "Columnwise"
#define MagmaRowwiseStr    "Rowwise"

#define MagmaNoVecStr        "NoVec"
#define MagmaVecStr          "Vec"
#define MagmaIVecStr         "IVec"
#define MagmaAllVecStr       "All"
#define MagmaSomeVecStr      "Some"
#define MagmaOverwriteVecStr "Overwrite"

#ifdef __cplusplus
extern "C" {
#endif

// --------------------
// translators
magma_trans_t  magma_trans_const ( char lapack_char );
magma_uplo_t   magma_uplo_const  ( char lapack_char );
magma_diag_t   magma_diag_const  ( char lapack_char );
magma_side_t   magma_side_const  ( char lapack_char );
magma_norm_t   magma_norm_const  ( char lapack_char );
magma_dist_t   magma_dist_const  ( char lapack_char );
magma_pack_t   magma_pack_const  ( char lapack_char );
magma_vec_t    magma_vec_const   ( char lapack_char );
magma_direct_t magma_direct_const( char lapack_char );
magma_storev_t magma_storev_const( char lapack_char );

char        lapacke_const( int magma_const );
const char* lapack_const ( int magma_const );

#define lapacke_order_const(c) lapack_const(c)
#define lapacke_trans_const(c) lapack_const(c)
#define lapacke_side_const( c) lapack_const(c)
#define lapacke_diag_const( c) lapack_const(c)
#define lapacke_uplo_const( c) lapack_const(c)

#define lapack_order_const(c)  lapack_const(c)
#define lapack_trans_const(c)  lapack_const(c)
#define lapack_side_const( c)  lapack_const(c)
#define lapack_diag_const( c)  lapack_const(c)
#define lapack_uplo_const( c)  lapack_const(c)

#ifdef HAVE_clAmdBlas
int                  amdblas_const      ( int           magma_const );
clAmdBlasOrder       amdblas_order_const( magma_order_t magma_const );
clAmdBlasTranspose   amdblas_trans_const( magma_trans_t magma_const );
clAmdBlasSide        amdblas_side_const ( magma_side_t  magma_const );
clAmdBlasDiag        amdblas_diag_const ( magma_diag_t  magma_const );
clAmdBlasUplo        amdblas_uplo_const ( magma_uplo_t  magma_const );
#endif

#ifdef CUBLAS_V2_H_
int                  cublas_const       ( int           magma_const );
cublasOperation_t    cublas_trans_const ( magma_trans_t magma_const );
cublasSideMode_t     cublas_side_const  ( magma_side_t  magma_const );
cublasDiagType_t     cublas_diag_const  ( magma_diag_t  magma_const );
cublasFillMode_t     cublas_uplo_const  ( magma_uplo_t  magma_const );
#endif

#ifdef HAVE_CBLAS
#include "cblas.h"
enum CBLAS_ORDER     cblas_order_const  ( magma_order_t magma_const );
enum CBLAS_TRANSPOSE cblas_trans_const  ( magma_trans_t magma_const );
enum CBLAS_SIDE      cblas_side_const   ( magma_side_t  magma_const );
enum CBLAS_DIAG      cblas_diag_const   ( magma_diag_t  magma_const );
enum CBLAS_UPLO      cblas_uplo_const   ( magma_uplo_t  magma_const );
#endif

// todo: above functions should all be inlined or macros for
// efficiency. Here's an example.
// In C99, static inline potentially wastes some space by
// emitting multiple definitions, but is portable.
static inline int cblas_const( int magma_const ) {
    assert( magma_const >= MagmaMinConst );
    assert( magma_const <= MagmaMaxConst );
    return magma_const;
}

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_TYPES_H
