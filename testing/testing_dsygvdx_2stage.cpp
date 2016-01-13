/*
    -- MAGMA (version 2.0.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2016

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @generated from testing/testing_zhegvdx_2stage.cpp normal z -> d, Wed Jan  6 17:59:50 2016

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
#include "magma_dbulge.h"
#include "magma_threadsetting.h"

#define REAL


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dsygvdx
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t gpu_time;

    double *h_A, *h_R, *h_B, *h_S, *h_work;

    #ifdef COMPLEX
    double *rwork;
    magma_int_t lrwork;
    #endif

    /* Matrix size */
    double *w1, *w2, result[2]={0,0};
    magma_int_t *iwork;
    magma_int_t N, n2, info, lwork, liwork;
    double c_zero    = MAGMA_D_ZERO;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");

    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;

    if ( opts.check && opts.jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVec;
    }

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% itype = %d, jobz = %s, range = %s, uplo = %s, fraction = %6.4f, ngpu = %d\n",
           int(opts.itype), lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           opts.fraction, int(abs_ngpu) );

    printf("%%   N     M   GPU Time (sec)\n");
    printf("%%===========================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2     = N*N;
            magma_dsyevdx_getworksize(N, threads, (opts.jobz == MagmaVec), 
                                     &lwork, 
                                     #ifdef COMPLEX
                                     &lrwork, 
                                     #endif
                                     &liwork);
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( h_A,    double, n2 );
            TESTING_MALLOC_CPU( h_B,    double, n2 );
            TESTING_MALLOC_CPU( w1,     double, N );
            TESTING_MALLOC_CPU( w2,     double, N );
            TESTING_MALLOC_CPU( iwork,  magma_int_t, liwork );
            
            TESTING_MALLOC_PIN( h_R,    double, n2 );
            TESTING_MALLOC_PIN( h_S,    double, n2 );
            TESTING_MALLOC_PIN( h_work, double, lwork );
            #ifdef COMPLEX
            TESTING_MALLOC_PIN( rwork,  double, lrwork);
            #endif

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlarnv( &ione, ISEED, &n2, h_B );
            magma_dmake_hpd( N, h_B, N );
            magma_dmake_symmetric( N, h_A, N );

            magma_int_t m1 = 0;
            double vl = 0;
            double vu = 0;
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

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================
            lapackf77_dlacpy( MagmaFullStr, &N, &N, h_A, &N, h_R, &N );
            lapackf77_dlacpy( MagmaFullStr, &N, &N, h_B, &N, h_S, &N );

            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                magma_dsygvdx_2stage( opts.itype, opts.jobz, range, opts.uplo,
                                      N, h_R, N, h_S, N, vl, vu, il, iu, &m1, w1,
                                      h_work, lwork,
                                      #ifdef COMPLEX
                                      rwork, lrwork,
                                      #endif
                                      iwork, liwork,
                                      &info );
            }
            else {
                magma_dsygvdx_2stage_m( abs_ngpu, opts.itype, opts.jobz, range, opts.uplo,
                                        N, h_R, N, h_S, N, vl, vu, il, iu, &m1, w1,
                                        h_work, lwork,
                                        #ifdef COMPLEX
                                        rwork, lrwork,
                                        #endif
                                        iwork, liwork,
                                        &info );
            }
            gpu_time = magma_wtime() - gpu_time;

            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvdx routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A| |Z| N )  (itype = 1)
                          | A B Z - Z D | / ( |A| |Z| N )  (itype = 2)
                          | B A Z - Z D | / ( |A| |Z| N )  (itype = 3)
                   (2)    | S(with V) - S(w/o V) | / | S |
                   =================================================================== */
                #ifdef REAL
                double *rwork = h_work + N*N;
                #endif

                result[0] = 1.;
                result[0] /= safe_lapackf77_dlansy("1", lapack_uplo_const(opts.uplo), &N, h_A, &N, rwork);
                result[0] /= lapackf77_dlange("1", &N, &m1, h_R, &N, rwork);

                if (opts.itype == 1) {
                    blasf77_dsymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &N, h_R, &N, &c_zero, h_work, &N);
                    for (int i=0; i < m1; ++i)
                        blasf77_dscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_neg_one, h_B, &N, h_R, &N, &c_one, h_work, &N);
                    result[0] *= lapackf77_dlange("1", &N, &m1, h_work, &N, rwork)/N;
                }
                else if (opts.itype == 2) {
                    blasf77_dsymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_B, &N, h_R, &N, &c_zero, h_work, &N);
                    for (int i=0; i < m1; ++i)
                        blasf77_dscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_dlange("1", &N, &m1, h_R, &N, rwork)/N;
                }
                else if (opts.itype == 3) {
                    blasf77_dsymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &N, h_R, &N, &c_zero, h_work, &N);
                    for (int i=0; i < m1; ++i)
                        blasf77_dscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_B, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_dlange("1", &N, &m1, h_R, &N, rwork)/N;
                }

                lapackf77_dlacpy( MagmaFullStr, &N, &N, h_A, &N, h_R, &N );
                lapackf77_dlacpy( MagmaFullStr, &N, &N, h_B, &N, h_S, &N );

                magma_int_t m2 = m1;
                lapackf77_dsygvd(&opts.itype, "N", lapack_uplo_const(opts.uplo), &N,
                              h_R, &N, h_S, &N, w2,
                              h_work, &lwork,
                              #ifdef COMPLEX
                              rwork, &lrwork,
                              #endif
                              iwork, &liwork,
                              &info);

                double maxw=0, diff=0;
                for (int j=0; j < m2; j++) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[1] = diff / (m2*maxw);
            }

            /* =====================================================================
               Print execution time
               =================================================================== */
            printf("%5d %5d   %7.2f\n",
                   (int) N, (int) m1, gpu_time);
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if (opts.itype == 1) {
                    printf("    | A Z - B Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed"));
                }
                else if (opts.itype == 2) {
                    printf("    | A B Z - Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed"));
                }
                else if (opts.itype == 3) {
                    printf("    | B A Z - Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed"));
                }
                printf(    "    | D(w/ Z) - D(w/o Z) | / |D|  = %8.2e   %s\n\n", result[1], (result[1] < tolulp ? "ok" : "failed"));
                status += ! (result[0] < tol && result[1] < tolulp);
            }

            TESTING_FREE_CPU( h_A   );
            TESTING_FREE_CPU( h_B   );
            TESTING_FREE_CPU( w1    );
            TESTING_FREE_CPU( w2    );
            TESTING_FREE_CPU( iwork );
            
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_PIN( h_S );
            TESTING_FREE_PIN( h_work );
            #ifdef COMPLEX
            TESTING_FREE_PIN( rwork );
            #endif
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
