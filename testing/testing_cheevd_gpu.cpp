/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Raffaele Solca
       @author Azzam Haidar
       @author Stan Tomov
       @author Mark Gates

       @generated from testing/testing_zheevd_gpu.cpp normal z -> c, Mon May  2 23:31:18 2016

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cheevd_gpu
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    /* Constants */
    const float d_zero = 0;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;
    
    /* Local variables */
    real_Double_t   gpu_time, cpu_time;
    magmaFloatComplex *h_A, *h_R, *h_Z, *h_work, aux_work[1];
    magmaFloatComplex_ptr d_R, d_Z;
    #ifdef COMPLEX
    float *rwork, aux_rwork[1];
    magma_int_t lrwork;
    #endif
    float *w1, *w2, result[4]={0, 0, 0, 0}, eps, abstol;
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, n2, info, lwork, liwork, lda, ldda;
    magma_int_t ISEED[4] = {0,0,0,1};
    eps = lapackf77_slamch( "E" );
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == MagmaNoVec);
    
    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;
    
    #ifdef REAL
    if (opts.version == 3 || opts.version == 4) {
        printf("%% magma_cheevr and magma_cheevx not available for real precisions (single, float).\n");
        return status;
    }
    #endif
    
    float tol    = opts.tolerance * lapackf77_slamch("E");
    float tolulp = opts.tolerance * lapackf77_slamch("P");
    
    printf("%% jobz = %s, range = %s, uplo = %s, fraction = %6.4f\n",
           lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           opts.fraction );

    printf("%%   N   CPU Time (sec)   GPU Time (sec)   |S-S_magma|   |A-USU^H|   |I-U^H U|\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2   = N*N;
            lda  = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            abstol = 0;  // auto, in cheevr
            
            // TODO: test vl-vu range
            magma_int_t m1 = 0;
            float vl = 0;
            float vu = 0;
            magma_int_t il = 0;
            magma_int_t iu = 0;
            if (opts.fraction == 0) {
                il = max( 1, magma_int_t(0.1*N) );
                iu = max( 1, magma_int_t(0.3*N) );
            }
            else {
                il = 1;
                iu = max( 1, magma_int_t(opts.fraction*N) );
            }

            // query for workspace sizes
            if ( opts.version == 1 || opts.version == 2 ) {
                magma_cheevd_gpu( opts.jobz, opts.uplo,
                                  N, NULL, ldda, NULL,  // A, w
                                  NULL, lda,            // host A
                                  aux_work,  -1,
                                  #ifdef COMPLEX
                                  aux_rwork, -1,
                                  #endif
                                  aux_iwork, -1,
                                  &info );
            }
            else if ( opts.version == 3 ) {
                #ifdef COMPLEX
                magma_cheevr_gpu( opts.jobz, range, opts.uplo,
                                  N, NULL, ldda,     // A
                                  vl, vu, il, iu, abstol,
                                  &m1, NULL,         // w
                                  NULL, ldda, NULL,  // Z, isuppz
                                  NULL, lda,         // host A
                                  NULL, lda,         // host Z
                                  aux_work,  -1,
                                  #ifdef COMPLEX
                                  aux_rwork, -1,
                                  #endif
                                  aux_iwork, -1,
                                  &info );
                #endif
            }
            else if ( opts.version == 4 ) {
                #ifdef COMPLEX
                magma_cheevx_gpu( opts.jobz, range, opts.uplo,
                                  N, NULL, ldda,     // A
                                  vl, vu, il, iu, abstol,
                                  &m1, NULL,         // w
                                  NULL, ldda,        // Z
                                  NULL, lda,         // host A
                                  NULL, lda,         // host Z
                                  aux_work,  -1,
                                  #ifdef COMPLEX
                                  aux_rwork,
                                  #endif
                                  aux_iwork,
                                  NULL,              // ifail
                                  &info );
                // cheevx doesn't query rwork, iwork; set them for consistency
                aux_rwork[0] = float(7*N);
                aux_iwork[0] = float(5*N);
                #endif
            }
            lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
            #ifdef COMPLEX
            lrwork = (magma_int_t) aux_rwork[0];
            #endif
            liwork = aux_iwork[0];
            
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, N*lda  );
            TESTING_MALLOC_CPU( w1,     float,             N      );
            TESTING_MALLOC_CPU( w2,     float,             N      );
            #ifdef COMPLEX
            TESTING_MALLOC_CPU( rwork,  float,             lrwork );
            #endif
            TESTING_MALLOC_CPU( iwork,  magma_int_t,        liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, N*lda  );
            TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lwork  );
            
            TESTING_MALLOC_DEV( d_R,    magmaFloatComplex, N*ldda );
            
            if (opts.version == 3) {
                TESTING_MALLOC_DEV( d_Z,    magmaFloatComplex, N*ldda     );
                TESTING_MALLOC_CPU( h_Z,    magmaFloatComplex, N*lda      );
                TESTING_MALLOC_CPU( isuppz, magma_int_t,        2*max(1,N) );
            }
            if (opts.version == 4) {
                TESTING_MALLOC_DEV( d_Z,    magmaFloatComplex, N*ldda     );
                TESTING_MALLOC_CPU( h_Z,    magmaFloatComplex, N*lda      );
                TESTING_MALLOC_CPU( ifail,  magma_int_t,        N          );
            }
            
            /* Clear eigenvalues, for |S-S_magma| check when fraction < 1. */
            lapackf77_slaset( "Full", &N, &ione, &d_zero, &d_zero, w1, &N );
            lapackf77_slaset( "Full", &N, &ione, &d_zero, &d_zero, w2, &N );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            magma_cmake_hermitian( N, h_A, lda );
            
            magma_csetmatrix( N, N, h_A, lda, d_R, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 1 ) {
                magma_cheevd_gpu( opts.jobz, opts.uplo,
                                  N, d_R, ldda, w1,
                                  h_R, lda,
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, lrwork,
                                  #endif
                                  iwork, liwork,
                                  &info );
            }
            else if ( opts.version == 2 ) {  // version 2: cheevdx computes selected eigenvalues/vectors
                magma_cheevdx_gpu( opts.jobz, range, opts.uplo,
                                   N, d_R, ldda,
                                   vl, vu, il, iu,
                                   &m1, w1,
                                   h_R, lda,
                                   h_work, lwork,
                                   #ifdef COMPLEX
                                   rwork, lrwork,
                                   #endif
                                   iwork, liwork,
                                   &info );
                //printf( "il %d, iu %d, m1 %d\n", il, iu, m1 );
            }
            else if ( opts.version == 3 ) {  // version 3: MRRR, computes selected eigenvalues/vectors
                // only complex version available
                #ifdef COMPLEX
                magma_cheevr_gpu( opts.jobz, range, opts.uplo,
                                  N, d_R, ldda,
                                  vl, vu, il, iu, abstol,
                                  &m1, w1,
                                  d_Z, ldda, isuppz,
                                  h_R, lda,  // host A
                                  h_Z, lda,  // host Z
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, lrwork,
                                  #endif
                                  iwork, liwork,
                                  &info );
                magmablas_clacpy( MagmaFull, N, N, d_Z, ldda, d_R, ldda, opts.queue );
                #endif
            }
            else if ( opts.version == 4 ) {  // version 3: cheevx (QR iteration), computes selected eigenvalues/vectors
                // only complex version available
                #ifdef COMPLEX
                magma_cheevx_gpu( opts.jobz, range, opts.uplo,
                                  N, d_R, ldda,
                                  vl, vu, il, iu, abstol,
                                  &m1, w1,
                                  d_Z, ldda,
                                  h_R, lda,
                                  h_Z, lda,
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, /*lrwork,*/
                                  #endif
                                  iwork, /*liwork,*/
                                  ifail,
                                  &info );
                magmablas_clacpy( MagmaFull, N, N, d_Z, ldda, d_R, ldda, opts.queue );
                #endif
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_cheevd_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            bool okay = true;
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U^H and the following 3 tests computed:
                   (1)    | A - U S U^H | / ( |A| N )
                   (2)    | I - U^H U   | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
                   =================================================================== */
                magma_cgetmatrix( N, N, d_R, ldda, h_R, lda, opts.queue );
                
                magmaFloatComplex *work;
                TESTING_MALLOC_CPU( work, magmaFloatComplex, 2*N*N );
                
                // e=NULL is unused since kband=0; tau=NULL is unused since itype=1
                lapackf77_chet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                  h_A, &lda,
                                  w1, NULL,
                                  h_R, &lda,
                                  h_R, &lda,
                                  NULL, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[0] );
                result[0] *= eps;
                result[1] *= eps;
                
                TESTING_FREE_CPU( work );  work=NULL;
                
                // Disable third eigenvalue check that calls routine again --
                // it obscures whether error occurs in first call above or in this call.
                // But see comparison to LAPACK below.
                //
                //magma_csetmatrix( N, N, h_A, lda, d_R, ldda, opts.queue );
                //magma_cheevd_gpu( MagmaNoVec, opts.uplo,
                //                  N, d_R, ldda, w2,
                //                  h_R, lda,
                //                  h_work, lwork,
                //                  #ifdef COMPLEX
                //                  rwork, lrwork,
                //                  #endif
                //                  iwork, liwork,
                //                  &info);
                //if (info != 0) {
                //    printf("magma_cheevd_gpu returned error %d: %s.\n",
                //           (int) info, magma_strerror( info ));
                //}
                //
                //float maxw=0, diff=0;
                //for( int j=0; j < N; j++ ) {
                //    maxw = max(maxw, fabs(w1[j]));
                //    maxw = max(maxw, fabs(w2[j]));
                //    diff = max(diff, fabs(w1[j] - w2[j]));
                //}
                //result[2] = diff / (N*maxw);
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                if ( opts.version == 1 || opts.version == 2 ) {
                    lapackf77_cheevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda, w2,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork, &lrwork,
                                      #endif
                                      iwork, &liwork,
                                      &info );
                }
                else if ( opts.version == 3 ) {
                    lapackf77_cheevr( lapack_vec_const(opts.jobz),
                                      lapack_range_const(range),
                                      lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda,
                                      &vl, &vu, &il, &iu, &abstol,
                                      &m1, w2,
                                      h_Z, &lda, isuppz,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork, &lrwork,
                                      #endif
                                      iwork, &liwork,
                                      &info );
                    lapackf77_clacpy( "Full", &N, &N, h_Z, &lda, h_A, &lda );
                }
                else if ( opts.version == 4 ) {
                    lapackf77_cheevx( lapack_vec_const(opts.jobz),
                                      lapack_range_const(range),
                                      lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda,
                                      &vl, &vu, &il, &iu, &abstol,
                                      &m1, w2,
                                      h_Z, &lda,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork,
                                      #endif
                                      iwork,
                                      ifail,
                                      &info );
                    lapackf77_clacpy( "Full", &N, &N, h_Z, &lda, h_A, &lda );
                }
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_cheevd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                // compare eigenvalues
                float maxw=0, diff=0;
                for( int j=0; j < N; j++ ) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[3] = diff / (N*maxw);
                
                okay = okay && (result[3] < tolulp);
                printf("%5d   %9.4f        %9.4f         %8.2e  ",
                       (int) N, cpu_time, gpu_time, result[3] );
            }
            else {
                printf("%5d      ---           %9.4f           ---     ",
                       (int) N, gpu_time);
            }
            
            // print error checks
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                okay = okay && (result[0] < tol) && (result[1] < tol);
                printf("    %8.2e    %8.2e", result[0], result[1] );
            }
            else {
                printf("      ---         ---   ");
            }
            printf("   %s\n", (okay ? "ok" : "failed"));
            status += ! okay;

            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( w1     );
            TESTING_FREE_CPU( w2     );
            #ifdef COMPLEX
            TESTING_FREE_CPU( rwork  );
            #endif
            TESTING_FREE_CPU( iwork  );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            
            TESTING_FREE_DEV( d_R );
            
            if ( opts.version == 3 ) {
                TESTING_FREE_DEV( d_Z    );
                TESTING_FREE_CPU( h_Z    );
                TESTING_FREE_CPU( isuppz );
            }
            if ( opts.version == 4 ) {
                TESTING_FREE_DEV( d_Z    );
                TESTING_FREE_CPU( h_Z    );
                TESTING_FREE_CPU( ifail  );
            }
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
