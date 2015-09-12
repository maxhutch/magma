/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from magma_zutil.cpp normal z -> d, Fri Sep 11 18:29:39 2015

       @author Mark Gates

       Utilities for testing.
*/

#include "testings.h"

#define REAL

#define A(i,j)  A[i + j*lda]

// --------------------
// Make a matrix symmetric/symmetric.
// Makes diagonal real.
// Sets Aji = conj( Aij ) for j < i, that is, copy & conjugate lower triangle to upper triangle.
extern "C"
void magma_dmake_symmetric( magma_int_t N, double* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_D_MAKE( MAGMA_D_REAL( A(i,i) ), 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = MAGMA_D_CNJG( A(i,j) );
        }
    }
}


// --------------------
// Make a matrix symmetric/symmetric positive definite.
// Increases diagonal by N, and makes it real.
// Sets Aji = conj( Aij ) for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_dmake_hpd( magma_int_t N, double* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_D_MAKE( MAGMA_D_REAL( A(i,i) ) + N, 0. );
        for( j=0; j < i; ++j ) {
            A(j,i) = MAGMA_D_CNJG( A(i,j) );
        }
    }
}

#ifdef COMPLEX
// --------------------
// Make a matrix real-symmetric
// Dose NOT make diagonal real.
// Sets Aji = Aij for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_dmake_symmetric( magma_int_t N, double* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        for( j=0; j < i; ++j ) {
            A(j,i) =  A(i,j);
        }
    }
}


// --------------------
// Make a matrix real-symmetric positive definite.
// Increases diagonal by N. Does NOT make diagonal real.
// Sets Aji = Aij for j < i, that is, copy lower triangle to upper triangle.
extern "C"
void magma_dmake_spd( magma_int_t N, double* A, magma_int_t lda )
{
    magma_int_t i, j;
    for( i=0; i < N; ++i ) {
        A(i,i) = MAGMA_D_MAKE( MAGMA_D_REAL( A(i,i) ) + N, MAGMA_D_IMAG( A(i,i) ) );
        for( j=0; j < i; ++j ) {
            A(j,i) = A(i,j);
        }
    }
}
#endif
