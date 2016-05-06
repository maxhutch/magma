/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/magma_zutil.cpp normal z -> c, Mon May  2 23:31:04 2016

       @author Mark Gates

       Utilities for testing.
*/

#include "testings.h"

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_threadsetting.h"  // to work around MKL bug

#define COMPLEX

#define A(i,j)  A[i + j*lda]

// --------------------
// Make a matrix symmetric/Hermitian.
// Makes diagonal real.
// Sets Aji = conj( Aij ) for j < i, that is, copy & conjugate lower triangle to upper triangle.
extern "C"
void magma_cmake_hermitian( magma_int_t N, magmaFloatComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_C_MAKE( MAGMA_C_REAL( A(i,i) ), 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = MAGMA_C_CONJ( A(i,j) );
        }
    }
}


// --------------------
// Make a matrix symmetric/Hermitian positive definite.
// Increases diagonal by N, and makes it real.
// Sets Aji = conj( Aij ) for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_cmake_hpd( magma_int_t N, magmaFloatComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_C_MAKE( MAGMA_C_REAL( A(i,i) ) + N, 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = MAGMA_C_CONJ( A(i,j) );
        }
    }
}

#ifdef COMPLEX
// --------------------
// Make a matrix complex-symmetric
// Dose NOT make diagonal real.
// Sets Aji = Aij for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_cmake_symmetric( magma_int_t N, magmaFloatComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        for( j=0; j < i; ++j ) {
            A(j,i) =  A(i,j);
        }
    }
}


// --------------------
// Make a matrix complex-symmetric positive definite.
// Increases diagonal by N. Does NOT make diagonal real.
// Sets Aji = Aij for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_cmake_spd( magma_int_t N, magmaFloatComplex* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_C_MAKE( MAGMA_C_REAL( A(i,i) ) + N, MAGMA_C_IMAG( A(i,i) ) );
        for( j=0; j < i; ++j ) {
            A(j,i) = A(i,j);
        }
    }
}
#endif


// --------------------
// MKL 11.1 has bug in multi-threaded clanhe; use single thread to work around.
// MKL 11.2 corrects it for inf, one, max norm.
// MKL 11.2 still segfaults for Frobenius norm.
// See testing_clanhe.cpp
float safe_lapackf77_clanhe(
    const char *norm, const char *uplo,
    const magma_int_t *n,
    const magmaFloatComplex *A, const magma_int_t *lda,
    float *work )
{
    #ifdef MAGMA_WITH_MKL
    // work around MKL bug in multi-threaded clanhe
    magma_int_t la_threads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads( 1 );
    #endif
    
    float result = lapackf77_clanhe( norm, uplo, n, A, lda, work );
    
    #ifdef MAGMA_WITH_MKL
    // end single thread to work around MKL bug
    magma_set_lapack_numthreads( la_threads );
    #endif
    
    return result;
}
