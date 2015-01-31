/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_ztrtri_diag.cpp normal z -> s, Fri Jan 30 19:00:23 2015
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define     h_A(i_, j_) (h_A     + (i_) + (j_)*lda)
#define h_dinvA(i_, j_) (h_dinvA + (i_) + (j_)*nb)


/* ////////////////////////////////////////////////////////////////////////////
   -- like axpy for matrices: B += alpha*A.
*/
void sgeadd(
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *A, magma_int_t lda,
    float       *B, magma_int_t ldb )
{
    #define A(i_, j_) (A + (i_) + (j_)*lda)
    #define B(i_, j_) (B + (i_) + (j_)*ldb)
    
    const magma_int_t ione = 1;
    
    for( int j=0; j < n; ++j ) {
        blasf77_saxpy( &m, &alpha, A(0,j), &ione, B(0,j), &ione );
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing strtri
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time=0;  //, cpu_perf=0, cpu_time=0;
    float          magma_error, norm_invA, work[1];
    magma_int_t N, lda, ldda, info;
    magma_int_t jb, nb, nblock, sizeA, size_inv;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    float *h_A, *h_dinvA;
    magmaFloat_ptr d_A, d_dinvA;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    const char *uplo_ = lapack_uplo_const(opts.uplo);

    // this is the NB hard coded into strtri_diag.
    nb = 128;
    
    printf("uplo = %s, diag = %s\n",
           lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag) );
    printf("    N  MAGMA Gflop/s (ms)   MAGMA error\n");
    printf("=======================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda = N;
            ldda = ((lda+31)/32)*32;
            nblock = (N+nb-1)/nb;
            gflops = nblock * FLOPS_STRTRI( nb ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,    float, lda*N );
            TESTING_MALLOC_CPU( ipiv,   magma_int_t,        N     );
            
            size_inv = nblock*nb*nb;
            TESTING_MALLOC_DEV( d_A,    float, ldda*N );
            TESTING_MALLOC_DEV( d_dinvA, float, size_inv );
            TESTING_MALLOC_CPU( h_dinvA, float, size_inv );
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            sizeA = lda*N;            
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_sgetrf( &N, &N, h_A, &lda, ipiv, &info );
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < j; ++i ) {
                    *h_A(i,j) = *h_A(j,i);
                }
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( N, N, h_A, lda, d_A, ldda );
            
            magma_time = magma_sync_wtime( NULL );
            magmablas_strtri_diag( opts.uplo, opts.diag, N, d_A, ldda, d_dinvA );
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_sgetvector( size_inv, d_dinvA, 1, h_dinvA, 1 );
            
            if ( opts.verbose ) {
                printf( "A%d=", (int) N );
                magma_sprint( N, N, h_A, lda );
                printf( "d_dinvA%d=", (int) N );
                magma_sprint( min(N+4, nb), min(N+4, nblock*nb), h_dinvA, nb );
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                //cpu_time = magma_wtime();
                lapackf77_strtri(
                    lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag),
                    &N, h_A, &lda, &info );
                //cpu_time = magma_wtime() - cpu_time;
                //cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                // |invA - invA_magma| / |invA|, accumulated over all diagonal blocks
                magma_error = 0;
                norm_invA   = 0;
                for( int i=0; i < N; i += nb ) {
                    jb = min( nb, N-i );
                    sgeadd( jb, jb, c_neg_one, h_A(i, i), lda, h_dinvA(0, i), nb );
                    magma_error = max( magma_error, lapackf77_slantr( "M", uplo_, MagmaNonUnitStr, &jb, &jb, h_dinvA(0, i), &nb,  work ));
                    norm_invA   = max( norm_invA,   lapackf77_slantr( "M", uplo_, MagmaNonUnitStr, &jb, &jb, h_A(i, i),     &lda, work ));
                }
                magma_error /= norm_invA;
                
                // CPU is doing N-by-N inverse, while GPU is doing (N/NB) NB-by-NB inverses.
                // So don't compare performance.
                printf("%5d   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int) N,
                        magma_perf,  1000.*magma_time,
                        //cpu_perf,    1000.*cpu_time,
                        magma_error,
                        (magma_error < tol ? "ok" : "failed"));
                status += ! (magma_error < tol);
            }
            else {
                printf("%5d   %7.2f (%7.2f)      ---\n",
                        (int) N,
                        magma_perf,  1000.*magma_time );
            }
            
            TESTING_FREE_CPU( h_A     );
            TESTING_FREE_CPU( ipiv    );
            
            TESTING_FREE_DEV( d_A     );
            TESTING_FREE_DEV( d_dinvA );
            TESTING_FREE_CPU( h_dinvA );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
