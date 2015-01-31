/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @precisions normal z -> c d s

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
#include "magma_zbulge.h"
#include "magma_threadsetting.h"

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvdx
*/
int main( int argc, char** argv)
{

    TESTING_INIT();

    real_Double_t   mgpu_time;
    magmaDoubleComplex *h_A, *h_Ainit, *h_B, *h_Binit, *h_work;

#if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    magma_int_t lrwork;
#endif

    double *w1, result=0;
    magma_int_t *iwork;
    magma_int_t N, n2, info, lwork, liwork;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;

    if ( opts.check && opts.jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVec;
    }

    printf("using: ngpu = %d, itype = %d, jobz = %s, range = %s, uplo = %s, opts.check = %d, fraction = %6.4f\n",
           (int) opts.ngpu, (int) opts.itype,
           lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           (int) opts.check, opts.fraction);
    
    printf("    N     M   ngpu   MGPU Time (sec)\n");
    printf("====================================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2     = N*N;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lwork  = magma_zbulge_get_lq2(N, threads) + 2*N + N*N;
            lrwork = 1 + 5*N +2*N*N;
            #else
            lwork  = magma_zbulge_get_lq2(N, threads) + 1 + 6*N + 2*N*N;
            #endif
            liwork = 3 + 5*N;


            //magma_int_t NB = 96;//magma_bulge_get_nb(N);
            //magma_int_t sizvblg = magma_zbulge_get_lq2(N, threads);        
            //magma_int_t siz = max(sizvblg,n2)+2*(N*NB+N)+24*N; 
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_PIN( h_A,    magmaDoubleComplex, n2 );
            TESTING_MALLOC_PIN( h_B,    magmaDoubleComplex, n2 );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_MALLOC_PIN( rwork,  double, lrwork);
            #endif

            TESTING_MALLOC_CPU( w1,     double, N );
            TESTING_MALLOC_CPU( iwork,  magma_int_t, liwork);
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlarnv( &ione, ISEED, &n2, h_B );
            magma_zmake_hpd( N, h_B, N );
            magma_zmake_hermitian( N, h_A, N );

            if ( opts.warmup || opts.check ) {
                TESTING_MALLOC_CPU( h_Ainit, magmaDoubleComplex, n2 );
                TESTING_MALLOC_CPU( h_Binit, magmaDoubleComplex, n2 );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_Ainit, &N );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_Binit, &N );
            }



            magma_int_t m1 = 0;
            double vl = 0;
            double vu = 0;
            magma_int_t il = 0;
            magma_int_t iu = 0;

            if (range == MagmaRangeI) {
                il = 1;
                iu = (int) (opts.fraction*N);
            }

            if ( opts.warmup ) {

                // ==================================================================
                // Warmup using MAGMA. I prefer to use smalltest to warmup A-
                // ==================================================================
                magma_zhegvdx_2stage_m(opts.ngpu, opts.itype, opts.jobz, range, opts.uplo,
                                       N, h_A, N, h_B, N, vl, vu, il, iu, &m1, w1,
                                       h_work, lwork,
                                       #if defined(PRECISION_z) || defined(PRECISION_c)
                                       rwork, lrwork,
                                       #endif
                                       iwork, liwork,
                                       &info);
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_Ainit, &N, h_A, &N );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_Binit, &N, h_B, &N );
            }

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================

            mgpu_time = magma_wtime();
            magma_zhegvdx_2stage_m(opts.ngpu, opts.itype, opts.jobz, range, opts.uplo,
                                   N, h_A, N, h_B, N, vl, vu, il, iu, &m1, w1,
                                   h_work, lwork,
                                       #if defined(PRECISION_z) || defined(PRECISION_c)
                                   rwork, lrwork,
                                       #endif
                                   iwork, liwork,
                                   &info);
            mgpu_time = magma_wtime() - mgpu_time;

            if ( opts.check && opts.jobz != MagmaNoVec ) {
                // ===================================================================
                // Check the results following the LAPACK's [zc]hegvdx routine.
                // A x = lambda B x is solved
                // and the following 3 tests computed:
                // (1)    | A Z - B Z D | / ( |A||Z| N )  (itype = 1)
                //        | A B Z - Z D | / ( |A||Z| N )  (itype = 2)
                //        | B A Z - Z D | / ( |A||Z| N )  (itype = 3)
                // ===================================================================
                #if defined(PRECISION_d) || defined(PRECISION_s)
                double *rwork = h_work + N*N;
                #endif
                result = 1.;
                result /= lapackf77_zlanhe("1", lapack_uplo_const(opts.uplo), &N, h_Ainit, &N, rwork);
                result /= lapackf77_zlange("1", &N , &m1, h_A, &N, rwork);

                if (opts.itype == 1) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_Ainit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_neg_one, h_Binit, &N, h_A, &N, &c_one, h_work, &N);
                    result *= lapackf77_zlange("1", &N, &m1, h_work, &N, rwork)/N;
                }
                else if (opts.itype == 2) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_Binit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_Ainit, &N, h_work, &N, &c_neg_one, h_A, &N);
                    result *= lapackf77_zlange("1", &N, &m1, h_A, &N, rwork)/N;
                }
                else if (opts.itype == 3) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_Ainit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_Binit, &N, h_work, &N, &c_neg_one, h_A, &N);
                    result *= lapackf77_zlange("1", &N, &m1, h_A, &N, rwork)/N;
                }
            }

            // ===================================================================
            // Print execution time
            // ===================================================================
            printf("%5d %5d   %4d   %7.2f\n",
                   (int) N, (int) m1, (int) opts.ngpu, mgpu_time);
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if (opts.itype==1) {
                    printf("    | A Z - B Z D | / (|A| |Z| N) = %8.2e   %s\n", result, (result < tol ? "ok" : "failed") );
                }
                else if (opts.itype==2) {
                    printf("    | A B Z - Z D | / (|A| |Z| N) = %8.2e   %s\n", result, (result < tol ? "ok" : "failed") );
                }
                else if (opts.itype==3) {
                    printf("    | B A Z - Z D | / (|A| |Z| N) = %8.2e   %s\n", result, (result < tol ? "ok" : "failed") );
                }
                printf("\n");
                status += ! (result < tol);
            }

            TESTING_FREE_PIN( h_A    );
            TESTING_FREE_PIN( h_B    );
            TESTING_FREE_PIN( h_work );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_FREE_PIN( rwork  );
            #endif

            TESTING_FREE_CPU( w1    );
            TESTING_FREE_CPU( iwork );
            if ( opts.warmup || opts.check ) {
                TESTING_FREE_CPU( h_Ainit );
                TESTING_FREE_CPU( h_Binit );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    /* Shutdown */
    TESTING_FINALIZE();
    return status;
}
