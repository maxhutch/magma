/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from magma_zutil.cpp normal z -> c, Fri Sep 11 18:29:39 2015

       @author Mark Gates

       Utilities for testing.
*/

#include "testings.h"

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
            A(j,i) = MAGMA_C_CNJG( A(i,j) );
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
            A(j,i) = MAGMA_C_CNJG( A(i,j) );
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
