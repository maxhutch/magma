/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @generated from testing_zhegvd.cpp normal z -> c, Fri Jan 30 19:00:26 2015

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chegvd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaFloatComplex *h_A, *h_R, *h_B, *h_S, *h_work;
    float *w1, *w2;
    float result[4] = {0, 0, 0, 0};
    magma_int_t *iwork;
    magma_int_t N, n2, info, nb, lwork, liwork, lda;
    magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    float d_one         =  1.;
    float d_neg_one     = -1.;
    #ifdef COMPLEX
    float *rwork;
    magma_int_t lrwork;
    #endif
    //float d_ten         = 10.;
    //magma_int_t izero    = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    float tol    = opts.tolerance * lapackf77_slamch("E");
    float tolulp = opts.tolerance * lapackf77_slamch("P");
    
    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == MagmaNoVec);
    
    printf("using: itype = %d, jobz = %s, uplo = %s\n",
           (int) opts.itype, lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo));

    printf("    N   CPU Time (sec)   GPU Time(sec)\n");
    printf("======================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = N*lda;
            nb     = magma_get_chetrd_nb(N);
            #ifdef COMPLEX
                lwork  = max( N + N*nb, 2*N + N*N );
                lrwork = 1 + 5*N +2*N*N;
            #else
                lwork  = max( 2*N + N*nb, 1 + 6*N + 2*N*N );
            #endif
            liwork = 3 + 5*N;

            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex,  n2     );
            TESTING_MALLOC_CPU( h_B,    magmaFloatComplex,  n2     );
            TESTING_MALLOC_CPU( w1,     float,              N      );
            TESTING_MALLOC_CPU( w2,     float,              N      );
            #ifdef COMPLEX
            TESTING_MALLOC_CPU( rwork,  float,              lrwork );
            #endif
            TESTING_MALLOC_CPU( iwork,  magma_int_t,         liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaFloatComplex,  n2     );
            TESTING_MALLOC_PIN( h_S,    magmaFloatComplex,  n2     );
            TESTING_MALLOC_PIN( h_work, magmaFloatComplex,  lwork  );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            //lapackf77_clatms( &N, &N, "U", ISEED, "P", w1, &five, &d_ten,
            //                 &d_one, &N, &N, lapack_uplo_const(opts.uplo), h_B, &lda, h_work, &info);
            //lapackf77_claset( "A", &N, &N, &c_zero, &c_one, h_B, &lda);
            lapackf77_clarnv( &ione, ISEED, &n2, h_B );
            magma_cmake_hpd( N, h_B, lda );
            lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            lapackf77_clacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
            
            /* warmup */
            if ( opts.warmup ) {
                magma_chegvd( opts.itype, opts.jobz, opts.uplo,
                              N, h_R, lda, h_S, lda, w1,
                              h_work, lwork,
                              #ifdef COMPLEX
                              rwork, lrwork,
                              #endif
                              iwork, liwork,
                              &info );
                if (info != 0)
                    printf("magma_chegvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                lapackf77_clacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_chegvd( opts.itype, opts.jobz, opts.uplo,
                          N, h_R, lda, h_S, lda, w1,
                          h_work, lwork,
                          #ifdef COMPLEX
                          rwork, lrwork,
                          #endif
                          iwork, liwork,
                          &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_chegvd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvd routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A||Z| N )   (itype = 1)
                          | A B Z - Z D | / ( |A||Z| N )   (itype = 2)
                          | B A Z - Z D | / ( |A||Z| N )   (itype = 3)
                   (2)    | I - V V' B | / ( N )           (itype = 1,2)
                          | B - V V' | / ( |B| N )         (itype = 3)
                   (3)    | D(with V) - D(w/o V) | / | D |
                   =================================================================== */
                //magmaFloatComplex *tau;
                
                #ifdef REAL
                float *rwork = h_work + N*N;
                #endif

                if ( opts.itype == 1 || opts.itype == 2 ) {
                    lapackf77_claset( "A", &N, &N, &c_zero, &c_one, h_S, &lda);
                    blasf77_cgemm("N", "C", &N, &N, &N, &c_one, h_R, &lda, h_R, &lda, &c_zero, h_work, &N);
                    blasf77_chemm("R", lapack_uplo_const(opts.uplo), &N, &N, &c_neg_one, h_B, &lda, h_work, &N, &c_one, h_S, &lda);
                    result[1] = lapackf77_clange("1", &N, &N, h_S, &lda, rwork) / N;
                }
                else if ( opts.itype == 3 ) {
                    lapackf77_clacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda);
                    blasf77_cherk(lapack_uplo_const(opts.uplo), "N", &N, &N, &d_neg_one, h_R, &lda, &d_one, h_S, &lda);
                    result[1] = lapackf77_clanhe("1", lapack_uplo_const(opts.uplo), &N, h_S, &lda, rwork) / N
                              / lapackf77_clanhe("1", lapack_uplo_const(opts.uplo), &N, h_B, &lda, rwork);
                }
                
                result[0] = 1.;
                result[0] /= lapackf77_clanhe("1", lapack_uplo_const(opts.uplo), &N, h_A, &lda, rwork);
                result[0] /= lapackf77_clange("1", &N, &N, h_R, &lda, rwork);
                
                if ( opts.itype == 1 ) {
                    blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for(int i=0; i < N; ++i)
                        blasf77_csscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_neg_one, h_B, &lda, h_R, &lda, &c_one, h_work, &N);
                    result[0] *= lapackf77_clange("1", &N, &N, h_work, &lda, rwork)/N;
                }
                else if ( opts.itype == 2 ) {
                    blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_B, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for(int i=0; i < N; ++i)
                        blasf77_csscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_A, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                    result[0] *= lapackf77_clange("1", &N, &N, h_R, &lda, rwork)/N;
                }
                else if ( opts.itype == 3 ) {
                    blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for(int i=0; i < N; ++i)
                        blasf77_csscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_chemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_B, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                    result[0] *= lapackf77_clange("1", &N, &N, h_R, &lda, rwork)/N;
                }
                
                /*
                assert( lwork >= 2*N*N );
                lapackf77_chet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                  h_A, &lda,
                                  w1, w1,
                                  h_R, &lda,
                                  h_R, &lda,
                                  tau, h_work, rwork, &result[0] );
                */
                
                // Disable eigenvalue check which calls routine again --
                // it obscures whether error occurs in first call above or in this call.
                // But see comparison to LAPACK below.
                //
                //lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                //lapackf77_clacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
                //
                //magma_chegvd( opts.itype, MagmaNoVec, opts.uplo,
                //              N, h_R, lda, h_S, lda, w2,
                //              h_work, lwork,
                //              #ifdef COMPLEX
                //              rwork, lrwork,
                //              #endif
                //              iwork, liwork,
                //              &info );
                //if (info != 0)
                //    printf("magma_chegvd returned error %d: %s.\n",
                //           (int) info, magma_strerror( info ));
                //
                //float maxw=0, diff=0;
                //for(int j=0; j < N; j++) {
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
                lapackf77_chegvd( &opts.itype, lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                  &N, h_A, &lda, h_B, &lda, w2,
                                  h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork, &lrwork,
                                  #endif
                                  iwork, &liwork,
                                  &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_chegvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // compare eigenvalues
                float maxw=0, diff=0;
                for(int j=0; j < N; j++) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[3] = diff / (N*maxw);
                
                printf("%5d     %7.2f         %7.2f\n",
                       (int) N, cpu_time, gpu_time);
            }
            else {
                printf("%5d       ---           %7.2f\n",
                       (int) N, gpu_time);
            }
            
            /* =====================================================================
               Print execution time
               =================================================================== */
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if ( opts.itype == 1 ) {
                    printf("    | A Z - B Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed") );
                }
                else if ( opts.itype == 2 ) {
                    printf("    | A B Z - Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed") );
                }
                else if ( opts.itype == 3 ) {
                    printf("    | B A Z - Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed") );
                }
                if ( opts.itype == 1 || opts.itype == 2 ) {
                    printf("    | I -   Z Z' B | /  N         = %8.2e   %s\n",   result[1], (result[1] < tol    ? "ok" : "failed") );
                }
                else {
                    printf("    | B -  Z Z' | / (|B| N)       = %8.2e   %s\n",   result[1], (result[1] < tol    ? "ok" : "failed") );
                }
                //printf(    "    | D(w/ Z) - D(w/o Z) | / |D|  = %8.2e   %s\n\n", result[2], (result[2] < tolulp ? "ok" : "failed") );
                status += ! (result[0] < tol && result[1] < tol);  // && result[2] < tolulp);
            }
            if ( opts.lapack ) {
                printf(    "    | D_magma - D_lapack | / |D|  = %8.2e   %s\n\n", result[3], (result[3] < tolulp ? "ok" : "failed") );
                status += ! (result[3] < tolulp);
            }
            
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_B    );
            TESTING_FREE_CPU( w1     );
            TESTING_FREE_CPU( w2     );
            #ifdef COMPLEX
            TESTING_FREE_CPU( rwork  );
            #endif
            TESTING_FREE_CPU( iwork  );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_S    );
            TESTING_FREE_PIN( h_work );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
