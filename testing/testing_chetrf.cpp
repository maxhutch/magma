/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zhetrf.cpp normal z -> c, Mon May  2 23:31:12 2016
       @author Ichitaro Yamazaki
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"  // for MAGMA_C_DIV
#include "testings.h"

/* ================================================================================================== */

// Initialize matrix to random & symmetrize. If nopiv, make positive definite.
// Having this in separate function ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix(
    bool nopiv,
    magma_int_t n, magmaFloatComplex *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_clarnv( &ione, ISEED, &n2, h_A );
    if (nopiv) {
        magma_cmake_hpd( n, h_A, lda );
    }
    else {
        magma_cmake_hermitian( n, h_A, lda );
    }
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
float get_residual(
    bool nopiv, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magma_int_t ione = 1;
    
    magma_int_t upper = (uplo == MagmaUpper);
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    magma_int_t i;
    magmaFloatComplex *x, *b;
    
    // initialize RHS
    TESTING_MALLOC_CPU( x, magmaFloatComplex, n );
    TESTING_MALLOC_CPU( b, magmaFloatComplex, n );
    lapackf77_clarnv( &ione, ISEED, &n, b );
    blasf77_ccopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    if (nopiv) {
        if (upper) {
            blasf77_ctrsm( MagmaLeftStr, MagmaUpperStr,
                           MagmaConjTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
            for (i=0; i < n; i++) {
                x[i] = MAGMA_C_DIV( x[i], A[i+i*lda] );
            }
            blasf77_ctrsm( MagmaLeftStr, MagmaUpperStr,
                           MagmaNoTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
        } else {
            blasf77_ctrsm( MagmaLeftStr, MagmaLowerStr,
                           MagmaNoTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
            for (i=0; i < n; i++) {
                x[i] = MAGMA_C_DIV( x[i], A[i+i*lda] );
            }
            blasf77_ctrsm( MagmaLeftStr, MagmaLowerStr,
                           MagmaConjTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
        }
    }
    else {
        lapackf77_chetrs( lapack_uplo_const(uplo), &n, &ione, A, &lda, ipiv, x, &n, &info );
    }
    if (info != 0) {
        printf("lapackf77_chetrs returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    }
    // reset to original A
    init_matrix( nopiv, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_cgemv( "Notrans", &n, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_clange( MagmaFullStr, &n, &n, A, &lda, work );
    norm_r = lapackf77_clange( MagmaFullStr, &n, &ione, b, &n, work );
    norm_x = lapackf77_clange( MagmaFullStr, &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_cprint( 1, n, b, 1 );
    
    TESTING_FREE_CPU( x );
    TESTING_FREE_CPU( b );
    
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%d\n", norm_r, norm_A, norm_x, n );
    return norm_r / (n * norm_A * norm_x);
}

float get_residual_aasen(
    bool nopiv, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    const magma_int_t ione = 1;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    magmaFloatComplex *L, *T;
    #define  A(i,j) ( A[(i) + (j)*lda])
    #define  L(i,j) ( L[(i) + (j)*n])
    #define  T(i,j) ( T[(i) + (j)*n])
    TESTING_MALLOC_CPU( L, magmaFloatComplex, n*n );
    TESTING_MALLOC_CPU( T, magmaFloatComplex, n*n );
    memset( L, 0, n*n*sizeof(magmaFloatComplex) );
    memset( T, 0, n*n*sizeof(magmaFloatComplex) );

    magma_int_t i, j, istart, piv;
    magma_int_t nb = magma_get_chetrf_aasen_nb(n);
    // extract T
    for (i=0; i < n; i++)
    {
        istart = max(0, i-nb);
        for (j=istart; j <= i; j++) {
            T(i,j) = A(i,j);
        }
        for (j=istart; j < i; j++) {
            T(j,i) = MAGMA_C_CONJ(A(i,j));
        }
    }
    // extract L
    for (i=0; i < min(n,nb); i++) 
    {
        L(i,i) = c_one;
    }
    for (i=nb; i < n; i++)
    {
        for (j=0; j < i-nb; j++) {
            L(i,nb+j) = A(i,j);
        }
        L(i,i) = c_one;
    }

    // solve
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    magmaFloatComplex *x, *b;
    
    // initialize RHS
    TESTING_MALLOC_CPU( x, magmaFloatComplex, n );
    TESTING_MALLOC_CPU( b, magmaFloatComplex, n );
    lapackf77_clarnv( &ione, ISEED, &n, b );
    blasf77_ccopy( &n, b, &ione, x, &ione );
    // pivot..
    for (i=0; i < n; i++) {
        piv = ipiv[i]-1;
        magmaFloatComplex val = x[i];
        x[i] = x[piv];
        x[piv] = val;
    }
    // forward solve
    blasf77_ctrsv( MagmaLowerStr, MagmaNoTransStr, MagmaUnitStr, &n, &L(0,0), &n, x, &ione );
    // banded solver
    magma_int_t nrhs = 1, *p = NULL;
    
    TESTING_MALLOC_CPU( p, magma_int_t, n );
    
    lapackf77_cgesv( &n, &nrhs, &T(0, 0), &n, p, x, &n, &info );
    
    TESTING_FREE_CPU( p );
    
    // backward solve
    blasf77_ctrsv( MagmaLowerStr, MagmaConjTransStr, MagmaUnitStr, &n, &L(0,0), &n, x, &ione );
    // pivot..
    for (i=n-1; i >= 0; i--) {
        piv = ipiv[i]-1;
        magmaFloatComplex val = x[i];
        x[i] = x[piv];
        x[piv] = val;
    }

    // reset to original A
    init_matrix( nopiv, n, A, lda );

    // compute r = Ax - b, saved in b
    blasf77_cgemv( "Notrans", &n, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_clange( MagmaFullStr, &n, &n, A, &lda, work );
    norm_r = lapackf77_clange( MagmaFullStr, &n, &ione, b, &n, work );
    norm_x = lapackf77_clange( MagmaFullStr, &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_cprint( 1, n, b, 1 );
    TESTING_FREE_CPU( L );
    TESTING_FREE_CPU( T );
    
    TESTING_FREE_CPU( x );
    TESTING_FREE_CPU( b );
    
    #undef T
    #undef L
    #undef A
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%d\n", norm_r, norm_A, norm_x, n );
    return norm_r / (n * norm_A * norm_x);
}

// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
float get_LDLt_error(
    bool nopiv, magma_uplo_t uplo, magma_int_t N,
    magmaFloatComplex *LD, magma_int_t lda,
    magma_int_t *ipiv)
{
    const magmaFloatComplex c_one  = MAGMA_C_ONE;
    const magmaFloatComplex c_zero = MAGMA_C_ZERO;
    
    magma_int_t i, j, piv;
    magmaFloatComplex *A, *L, *D;
    float work[1], matnorm, residual;
    
    #define LD(i,j) (LD[(i) + (j)*lda])
    #define  A(i,j) ( A[(i) + (j)*N])
    #define  L(i,j) ( L[(i) + (j)*N])
    #define  D(i,j) ( D[(i) + (j)*N])

    TESTING_MALLOC_CPU( A, magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( L, magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( D, magmaFloatComplex, N*N );
    memset( L, 0, N*N*sizeof(magmaFloatComplex) );
    memset( D, 0, N*N*sizeof(magmaFloatComplex) );

    // set to original A, and apply pivoting
    init_matrix( nopiv, N, A, N );
    if (uplo == MagmaUpper) {
        for (j=N-1; j >= 0; j--) {
            piv = (nopiv ? j+1 : ipiv[j]);
            if (piv < 0) {
                piv = -(piv+1);
                // extract 2-by-2 pivot
                D(j,j)     = LD(j,j);
                D(j,j-1)   = MAGMA_C_CONJ(LD(j-1,j));
                D(j-1,j)   = LD(j-1,j);
                D(j-1,j-1) = LD(j-1,j-1);
                // exract L
                L(j,j) = c_one;
                for (i=0; i < j-1; i++) {
                    L(i,j) = LD(i,j);
                }
                j--;
                L(j,j) = c_one;
                for (i=0; i < j; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=j+2; i < N; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            } else {
                piv = piv-1;
                // extract 1-by-1 pivot
                D(j,j) = LD(j,j);
                // exract L
                L(j,j) = c_one;
                for (i=0; i < j; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=j+1; i < N; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            }
        }
        if (nopiv) {
            // compute W = D*U
            blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                          &c_one, D, &N, L, &N, &c_zero, LD, &lda);
            // compute D = U^H*W
            blasf77_cgemm(MagmaConjTransStr, MagmaNoTransStr, &N, &N, &N,
                          &c_one, L, &N, LD, &lda, &c_zero, D, &N);
        } else {
            // compute W = U*D
            blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                          &c_one, L, &N, D, &N, &c_zero, LD, &lda);
            // compute D = W*U^H
            blasf77_cgemm(MagmaNoTransStr, MagmaConjTransStr, &N, &N, &N,
                          &c_one, LD, &lda, L, &N, &c_zero, D, &N);
        }
    } else {
        for (j=0; j < N; j++) {
            piv = (nopiv ? j+1 : ipiv[j]);
            if (piv < 0) {
                piv = -(piv+1);
                // extract 2-by-2 pivot
                D(j,j)     = LD(j,j);
                D(j,j+1)   = MAGMA_C_CONJ(LD(j+1,j));
                D(j+1,j)   = LD(j+1,j);
                D(j+1,j+1) = LD(j+1,j+1);
                // exract L
                L(j,j) = c_one;
                for (i=j+2; i < N; i++) {
                    L(i,j) = LD(i,j);
                }
                j++;
                L(j,j) = c_one;
                for (i=j+1; i < N; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=0; i < j-1; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            } else {
                piv = piv-1;
                // extract 1-by-1 pivot
                D(j,j) = LD(j,j);
                // exract L
                L(j,j) = c_one;
                for (i=j+1; i < N; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=0; i < j; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            }
        }
        // compute W = L*D
        blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                      &c_one, L, &N, D, &N, &c_zero, LD, &lda);
        // compute D = W*L^H
        blasf77_cgemm(MagmaNoTransStr, MagmaConjTransStr, &N, &N, &N,
                      &c_one, LD, &lda, L, &N, &c_zero, D, &N);
    }
    // compute norm of A
    matnorm = lapackf77_clange(MagmaFullStr, &N, &N, A, &lda, work);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < N; i++ ) {
            D(i,j) = MAGMA_C_SUB( D(i,j), A(i,j) );
        }
    }
    residual = lapackf77_clange(MagmaFullStr, &N, &N, D, &N, work);

    TESTING_FREE_CPU( A );
    TESTING_FREE_CPU( L );
    TESTING_FREE_CPU( D );

    return residual / (matnorm * N);
}


float get_LTLt_error(
    bool nopiv, magma_uplo_t uplo, magma_int_t N,
    magmaFloatComplex *LT, magma_int_t lda,
    magma_int_t *ipiv)
{
    float work[1], matnorm, residual;
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex *A, *L, *T;
    
    #define LT(i,j) (LT[(i) + (j)*lda])
    #define  T(i,j) ( T[(i) + (j)*N])
    
    TESTING_MALLOC_CPU( A, magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( L, magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( T, magmaFloatComplex, N*N );
    memset( L, 0, N*N*sizeof(magmaFloatComplex) );
    memset( T, 0, N*N*sizeof(magmaFloatComplex) );

    magma_int_t i, j, istart, piv;
    magma_int_t nb = magma_get_chetrf_aasen_nb(N);
    
    // for debuging
    /*
    magma_int_t *p;
    TESTING_MALLOC_CPU( p, magma_int_t, n );
    for (i=0; i < N; i++) {
        p[i] = i;
    }
    for (i=0; i < N; i++) {
        piv = ipiv[i]-1;
        i2 = p[piv];
        p[piv] = p[i];
        p[i] = i2;
    }
    printf( " p=[" );
    for (i=0; i < N; i++) {
        printf("%d ", p[i] );
    }
    printf( "];\n" );
    TESTING_FREE_CPU( p );
    */
    
    // extract T
    for (i=0; i < N; i++) {
        istart = max(0, i-nb);
        for (j=istart; j <= i; j++) {
            T(i,j) = LT(i,j);
        }
        for (j=istart; j < i; j++) {
            T(j,i) = MAGMA_C_CONJ( LT(i,j) );
        }
    }
    //printf( "T=" );
    //magma_cprint(N,N, &T(0,0),N);
    // extract L
    for (i=0; i < min(N,nb); i++) 
    {
        L(i,i) = c_one;
    }
    for (i=nb; i < N; i++)
    {
        for (j=0; j < i-nb; j++) {
            L(i,nb+j) = LT(i,j);
        }
        L(i,i) = c_one;
    }
    //printf( "L=" );
    //magma_cprint(N,N, &L(0,0),N);

    // compute LD = L*T
    blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                  &c_one, L, &N, T, &N, &c_zero, LT, &lda);
    // compute T = LD*L^H
    blasf77_cgemm(MagmaNoTransStr, MagmaConjTransStr, &N, &N, &N,
                  &c_one, LT, &lda, L, &N, &c_zero, T, &N);

    // compute norm of A
    init_matrix( nopiv, N, A, N );
    matnorm = lapackf77_clange(MagmaFullStr, &N, &N, A, &lda, work);
    //printf( "A0=" );
    //magma_cprint(N,N, &A(0,0),N);

    // apply symmetric pivoting
    for (j=0; j < N; j++) {
        piv = ipiv[j]-1;
        if (piv != j) {
            // apply row-pivoting to A
            for (i=0; i < N; i++) {
                magmaFloatComplex val = A(j,i);
                A(j,i) = A(piv,i);
                A(piv,i) = val;
            }
            // apply col-pivoting to A
            for (i=0; i < N; i++) {
                magmaFloatComplex val = A(i,j);
                A(i,j) = A(i,piv);
                A(i,piv) = val;
            }
        }
    }

    // compute factorization error
    for(j = 0; j < N; j++ ) {
        for(i = 0; i < N; i++ ) {
            T(i,j) = MAGMA_C_SUB( T(i,j), A(i,j) );
        }
    }
    residual = lapackf77_clange(MagmaFullStr, &N, &N, T, &N, work);

    TESTING_FREE_CPU( A );
    TESTING_FREE_CPU( L );
    TESTING_FREE_CPU( T );

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chetrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magmaFloatComplex *h_A, *work, temp;
    real_Double_t   gflops, gpu_perf, gpu_time = 0.0, cpu_perf=0, cpu_time=0;
    float          error, error_lapack = 0.0;
    magma_int_t     *ipiv;
    magma_int_t     i, cpu_panel = 1, N, n2, lda, lwork, info;
    magma_int_t     status = 0;
    magma_int_t     cpu = 0, nopiv = 0, nopiv_gpu = 0, row = 0, aasen = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // TODO: this doesn't work. Options need to be added to parse_opts()
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--cpu-panel", argv[i]) == 0) cpu_panel = 1;
        if ( strcmp("--gpu-panel", argv[i]) == 0) cpu_panel = 0;
    }
    
    switch (opts.version) {
        case 1:
            cpu = 1;
            printf( "\n%% CPU-Interface to Bunch-Kauffman on GPU" );
            break;
        case 2:
            //gpu = 1;
            printf( "\n%% GPU-Interface to Bunch-Kauffman on GPU" );
            printf( "\n not yet..\n\n" );
            return 0;
            break;
        case 3:
            nopiv = 1;
            printf( "\n%% CPU-Interface to hybrid Non-pivoted LDLt (A is SPD)" );
            break;
        case 4:
            nopiv_gpu = 1;
            printf( "\n%% GPU-Interface to hybrid Non-pivoted LDLt (A is SPD)" );
            break;
        //case 5:
        //    row = 1;
        //    printf( "\n Bunch-Kauffman: GPU-only version (row-major)" );
        //    break;
        case 6:
            aasen = 1;
            printf( "\n%% CPU-Interface to Aasen's (%s)",(cpu_panel ? "CPU panel" : "GPU panel") );
            break;
        default:
            printf( "\nversion = %d not supported\n\n", (int) opts.version );
            return 0;
    }
    printf( " (%s)\n", lapack_uplo_const(opts.uplo) );
    printf( " (--version: 1 = Bunch-Kauffman (CPU), 2 = Bunch-Kauffman (GPU), 3 = No-piv (CPU), 4 = No-piv (GPU))\n\n" );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    if ( opts.check == 2 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |PAP^H - LDL^H|/(N*|A|)\n");
    }
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            gflops = FLOPS_CPOTRF( N ) / 1e9;
            
            TESTING_MALLOC_PIN( ipiv, magma_int_t, N );
            TESTING_MALLOC_PIN( h_A,  magmaFloatComplex, n2 );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                lwork = -1;
                lapackf77_chetrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, ipiv, &temp, &lwork, &info );
                lwork = (magma_int_t)MAGMA_C_REAL( temp );
                TESTING_MALLOC_CPU( work, magmaFloatComplex, lwork );

                init_matrix( nopiv, N, h_A, lda );
                cpu_time = magma_wtime();
                lapackf77_chetrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, ipiv, work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_chetrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                error_lapack = get_residual( nopiv, opts.uplo, N, h_A, lda, ipiv );

                TESTING_FREE_CPU( work );
            }
           
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( (nopiv | nopiv_gpu), N, h_A, lda );

            //printf( "A0=" );
            //magma_cprint(N,N,h_A,lda);
            if (nopiv) {
                // CPU-interface to non-piv LDLt
                gpu_time = magma_wtime();
                magma_chetrf_nopiv( opts.uplo, N, h_A, lda, &info);
                gpu_time = magma_wtime() - gpu_time;
            } else if (cpu) {
                // CPU-interface to Bunch-Kauffman LDLt
                gpu_time = magma_wtime();
                magma_chetrf( opts.uplo, N, h_A, lda, ipiv, &info);
                gpu_time = magma_wtime() - gpu_time;
            } else if (nopiv_gpu) {
                // GPU-interface to non-piv LDLt
                magma_int_t ldda = magma_roundup( N, opts.align );
                magmaFloatComplex_ptr d_A;
                TESTING_MALLOC_DEV( d_A, magmaFloatComplex, N*ldda );
                magma_csetmatrix(N, N, h_A, lda, d_A, ldda, opts.queue );
                gpu_time = magma_wtime();
                magma_chetrf_nopiv_gpu( opts.uplo, N, d_A, ldda, &info);
                gpu_time = magma_wtime() - gpu_time;
                magma_cgetmatrix(N, N, d_A, ldda, h_A, lda, opts.queue );
                TESTING_FREE_DEV( d_A );
            } else if (aasen) {
                // CPU-interface to Aasen's LTLt
                gpu_time = magma_wtime();
                magma_chetrf_aasen( opts.uplo, cpu_panel, N, h_A, lda, ipiv, &info);
                gpu_time = magma_wtime() - gpu_time;
            } else if (row) {
                //magma_chetrf_gpu_row( opts.uplo, N, h_A, lda, ipiv, work, lwork, &info);
            } else {
                //magma_chetrf_hybrid( opts.uplo, N, h_A, lda, ipiv, work, lwork, &info);
            }
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_chetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) N, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) N, (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 2 ) {
                if (aasen) {
                    error = get_residual_aasen( (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                } else {
                    error = get_residual( (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                }
                printf("   %8.2e   %s", error, (error < tol ? "ok" : "failed"));
                if (opts.lapack)
                    printf(" (lapack rel.res. = %8.2e)", error_lapack);
                printf("\n");
                status += ! (error < tol);
            }
            else if ( opts.check ) {
                if (aasen) {
                    error = get_LTLt_error( (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                } else {
                    error = get_LDLt_error( (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                }
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("     ---   \n");
            }
 
            TESTING_FREE_PIN( ipiv );
            TESTING_FREE_PIN( h_A  );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
