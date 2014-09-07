/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

    @author Raffaele Solca
    @author Azzam Haidar

    @precisions normal z -> c d s

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvdx
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time /*cpu_time*/;
    magmaDoubleComplex *h_A, *h_R, *h_B, *h_S, *h_work;
    double *w1, *w2, vl=0, vu=0;
    double result[2] = {0};
    magma_int_t *iwork;
    magma_int_t N, n2, info, il, iu, m1, m2, nb, lwork, liwork;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    #if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    magma_int_t lrwork;
    #endif
    //double d_one         =  1.;
    //double d_ten         = 10.;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");
    
    if ( opts.check && opts.jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVec;
    }
    
    printf("using: itype = %d, jobz = %s, uplo = %s, check = %d, fraction = %6.4f\n",
           (int) opts.itype, lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (int) opts.check, opts.fraction);

    printf("    N     M   GPU Time (sec)\n");
    printf("============================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2     = N*N;
            nb     = magma_get_zhetrd_nb(N);
            #if defined(PRECISION_z) || defined(PRECISION_c)
                lwork  = max( N + N*nb, 2*N + N*N );
                lrwork = 1 + 5*N +2*N*N;
            #else
                lwork  = max( 2*N + N*nb, 1 + 6*N + 2*N*N );
            #endif
            liwork = 3 + 5*N;

            if ( opts.fraction == 0 ) {
                il = N / 10;
                iu = N / 5+il;
            }
            else {
                il = 1;
                iu = (int) (opts.fraction*N);
                if (iu < 1) iu = 1;
            }

            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_B,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( w1,     double,             N      );
            TESTING_MALLOC_CPU( w2,     double,             N      );
            TESTING_MALLOC_CPU( iwork,  magma_int_t,        liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_PIN( h_S,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork  );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_MALLOC_PIN( rwork, double, lrwork);
            #endif
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlarnv( &ione, ISEED, &n2, h_B );
            magma_zmake_hpd( N, h_B, N );
            magma_zmake_hermitian( N, h_A, N );

            // ==================================================================
            // Warmup using MAGMA
            // ==================================================================
            if(opts.warmup){
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
                
                magma_zhegvdx( opts.itype, opts.jobz, MagmaRangeI, opts.uplo,
                               N, h_R, N, h_S, N, vl, vu, il, iu, &m1, w1,
                               h_work, lwork,
                               #if defined(PRECISION_z) || defined(PRECISION_c)
                               rwork, lrwork,
                               #endif      
                               iwork, liwork,
                               &info );
                if (info != 0)
                    printf("magma_zhegvdx returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );

            gpu_time = magma_wtime();
            magma_zhegvdx( opts.itype, opts.jobz, MagmaRangeI, opts.uplo,
                           N, h_R, N, h_S, N, vl, vu, il, iu, &m1, w1,
                           h_work, lwork,
                           #if defined(PRECISION_z) || defined(PRECISION_c)
                           rwork, lrwork,
                           #endif
                           iwork, liwork,
                           &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zhegvdx returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvdx routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A||Z| N )  (itype = 1)
                          | A B Z - Z D | / ( |A||Z| N )  (itype = 2)
                          | B A Z - Z D | / ( |A||Z| N )  (itype = 3)
                   (2)    | S(with V) - S(w/o V) | / | S |
                   =================================================================== */
                #if defined(PRECISION_d) || defined(PRECISION_s)
                double *rwork = h_work + N*N;
                #endif
                double temp1, temp2;
                
                result[0] = 1.;
                result[0] /= lapackf77_zlanhe("1", lapack_uplo_const(opts.uplo), &N, h_A, &N, rwork);
                result[0] /= lapackf77_zlange("1", &N, &m1, h_R, &N, rwork);
                
                if (opts.itype == 1) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &N, h_R, &N, &c_zero, h_work, &N);
                    for(int i=0; i < m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_neg_one, h_B, &N, h_R, &N, &c_one, h_work, &N);
                    result[0] *= lapackf77_zlange("1", &N, &m1, h_work, &N, rwork)/N;
                }
                else if (opts.itype == 2) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_B, &N, h_R, &N, &c_zero, h_work, &N);
                    for(int i=0; i < m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_zlange("1", &N, &m1, h_R, &N, rwork)/N;
                }
                else if (opts.itype == 3) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &N, h_R, &N, &c_zero, h_work, &N);
                    for(int i=0; i < m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_B, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_zlange("1", &N, &m1, h_R, &N, rwork)/N;
                }
                
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
                
                magma_zhegvdx( opts.itype, MagmaNoVec, MagmaRangeI, opts.uplo,
                               N, h_R, N, h_S, N, vl, vu, il, iu, &m2, w2,
                               h_work, lwork,
                               #if defined(PRECISION_z) || defined(PRECISION_c)
                               rwork, lrwork,
                               #endif
                               iwork, liwork,
                               &info );
                if (info != 0)
                    printf("magma_zhegvdx returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                temp1 = temp2 = 0;
                for(int j=0; j < m2; j++) {
                    temp1 = max(temp1, fabs(w1[j]));
                    temp1 = max(temp1, fabs(w2[j]));
                    temp2 = max(temp2, fabs(w1[j]-w2[j]));
                }
                result[1] = temp2 / (((double)m2)*temp1);
            }
            
            /* =====================================================================
               Print execution time
               =================================================================== */
            printf("%5d %5d   %7.2f\n",
                   (int) N, (int) m1, gpu_time);
            if ( opts.check ) {
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if (opts.itype == 1) {
                    printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed"));
                }
                else if (opts.itype == 2) {
                    printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed"));
                }
                else if (opts.itype == 3) {
                    printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %8.2e   %s\n",   result[0], (result[0] < tol    ? "ok" : "failed"));
                }
                printf(    "(2)    | D(w/ Z) - D(w/o Z) | / |D|  = %8.2e   %s\n\n", result[1], (result[1] < tolulp ? "ok" : "failed"));
                status += ! (result[0] < tol && result[1] < tolulp);
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( w1  );
            TESTING_FREE_CPU( w2  );
            TESTING_FREE_CPU( iwork );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_S    );
            TESTING_FREE_PIN( h_work );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_FREE_PIN( rwork );
            #endif
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
