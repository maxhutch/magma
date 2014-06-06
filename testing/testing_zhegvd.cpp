/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    @author Raffaele Solca
    @author Azzam Haidar

    @precisions normal z -> c

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

#define absv(v1) ((v1)>0? (v1): -(v1))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *h_B, *h_S, *h_work;
    double *rwork, *w1, *w2;
    double result[4] = {0};
    magma_int_t *iwork;
    magma_int_t N, n2, info, nb, lwork, liwork, lda, lrwork;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double d_one         =  1.;
    double d_neg_one     = -1.;
    //double d_ten         = 10.;
    //magma_int_t izero    = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");
    
    if ( opts.check && opts.jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVec;
    }
    
    printf("    N   CPU Time (sec)   GPU Time(sec)\n");
    printf("======================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            n2     = N*lda;
            nb     = magma_get_zhetrd_nb(N);
            lwork  = 2*N*nb + N*N;
            lrwork = 1 + 5*N +2*N*N;
            liwork = 3 + 5*N;

            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex,  n2     );
            TESTING_MALLOC_CPU( h_B,    magmaDoubleComplex,  n2     );
            TESTING_MALLOC_CPU( w1,     double,              N      );
            TESTING_MALLOC_CPU( w2,     double,              N      );
            TESTING_MALLOC_CPU( rwork,  double,              lrwork );
            TESTING_MALLOC_CPU( iwork,  magma_int_t,         liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex,  n2     );
            TESTING_MALLOC_PIN( h_S,    magmaDoubleComplex,  n2     );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex,  lwork  );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            //lapackf77_zlatms( &N, &N, "U", ISEED, "P", w1, &five, &d_ten,
            //                 &d_one, &N, &N, &opts.uplo, h_B, &lda, h_work, &info);
            //lapackf77_zlaset( "A", &N, &N, &c_zero, &c_one, h_B, &lda);
            lapackf77_zlarnv( &ione, ISEED, &n2, h_B );
            magma_zmake_hpd( N, h_B, lda );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &lda, h_S, &lda );
            
            /* warmup */
            if ( opts.warmup ) {
                magma_zhegvd( opts.itype, opts.jobz, opts.uplo,
                              N, h_R, lda, h_S, lda, w1,
                              h_work, lwork,
                              rwork, lrwork,
                              iwork, liwork,
                              &info );
                if (info != 0)
                    printf("magma_zhegvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &lda, h_S, &lda );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zhegvd( opts.itype, opts.jobz, opts.uplo,
                          N, h_R, lda, h_S, lda, w1,
                          h_work, lwork,
                          rwork, lrwork,
                          iwork, liwork,
                          &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zhegvd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvd routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A||Z| N )   (itype = 1)
                          | A B Z - Z D | / ( |A||Z| N )   (itype = 2)
                          | B A Z - Z D | / ( |A||Z| N )   (itype = 3)
                   (2)    | I - V V' B | / ( N )           (itype = 1,2)
                          | B - V V' | / ( |B| N )         (itype = 3)
                   (3)    | S(with V) - S(w/o V) | / | S |
                   =================================================================== */
                double temp1, temp2;
                //magmaDoubleComplex *tau;
                
                if ( opts.itype == 1 || opts.itype == 2 ) {
                    lapackf77_zlaset( "A", &N, &N, &c_zero, &c_one, h_S, &lda);
                    blasf77_zgemm("N", "C", &N, &N, &N, &c_one, h_R, &lda, h_R, &lda, &c_zero, h_work, &N);
                    blasf77_zhemm("R", &opts.uplo, &N, &N, &c_neg_one, h_B, &lda, h_work, &N, &c_one, h_S, &lda);
                    result[1] = lapackf77_zlange("1", &N, &N, h_S, &lda, rwork) / N;
                }
                else if ( opts.itype == 3 ) {
                    lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &lda, h_S, &lda);
                    blasf77_zherk(&opts.uplo, "N", &N, &N, &d_neg_one, h_R, &lda, &d_one, h_S, &lda);
                    result[1] = lapackf77_zlanhe("1", &opts.uplo, &N, h_S, &lda, rwork) / N
                              / lapackf77_zlanhe("1", &opts.uplo, &N, h_B, &lda, rwork);
                }
                
                result[0] = 1.;
                result[0] /= lapackf77_zlanhe("1", &opts.uplo, &N, h_A, &lda, rwork);
                result[0] /= lapackf77_zlange("1", &N, &N, h_R, &lda, rwork);
                
                if ( opts.itype == 1 ) {
                    blasf77_zhemm("L", &opts.uplo, &N, &N, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", &opts.uplo, &N, &N, &c_neg_one, h_B, &lda, h_R, &lda, &c_one, h_work, &N);
                    result[0] *= lapackf77_zlange("1", &N, &N, h_work, &lda, rwork)/N;
                }
                else if ( opts.itype == 2 ) {
                    blasf77_zhemm("L", &opts.uplo, &N, &N, &c_one, h_B, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", &opts.uplo, &N, &N, &c_one, h_A, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                    result[0] *= lapackf77_zlange("1", &N, &N, h_R, &lda, rwork)/N;
                }
                else if ( opts.itype == 3 ) {
                    blasf77_zhemm("L", &opts.uplo, &N, &N, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", &opts.uplo, &N, &N, &c_one, h_B, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                    result[0] *= lapackf77_zlange("1", &N, &N, h_R, &lda, rwork)/N;
                }
                
                /*
                lapackf77_zhet21( &ione, &opts.uplo, &N, &izero,
                                  h_A, &lda,
                                  w1, w1,
                                  h_R, &lda,
                                  h_R, &lda,
                                  tau, h_work, rwork, &result[0] );
                */
                
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &lda, h_S, &lda );
                
                magma_zhegvd( opts.itype, MagmaNoVec, opts.uplo,
                              N, h_R, lda, h_S, lda, w2,
                              h_work, lwork,
                              rwork, lrwork,
                              iwork, liwork,
                              &info );
                if (info != 0)
                    printf("magma_zhegvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                temp1 = temp2 = 0;
                for(int j=0; j<N; j++) {
                    temp1 = max(temp1, absv(w1[j]));
                    temp1 = max(temp1, absv(w2[j]));
                    temp2 = max(temp2, absv(w1[j]-w2[j]));
                }
                result[2] = temp2 / (((double)N)*temp1);
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zhegvd( &opts.itype, &opts.jobz, &opts.uplo,
                                  &N, h_A, &lda, h_B, &lda, w2,
                                  h_work, &lwork,
                                  rwork, &lrwork,
                                  iwork, &liwork,
                                  &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_zhegvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
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
            if ( opts.check ) {
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if ( opts.itype==1 )
                    printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %8.2e%s\n", result[0], (result[0] < tol ? "" : "  failed") );
                else if ( opts.itype==2 )
                    printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %8.2e%s\n", result[0], (result[0] < tol ? "" : "  failed") );
                else if ( opts.itype==3 )
                    printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %8.2e%s\n", result[0], (result[0] < tol ? "" : "  failed") );
                if ( opts.itype==1 || opts.itype==2 )
                    printf("(2)    | I -   Z Z' B | /  N         = %8.2e%s\n", result[1], (result[1] < tol ? "" : "  failed") );
                else
                    printf("(2)    | B -  Z Z' | / (|B| N)       = %8.2e%s\n", result[1], (result[1] < tol ? "" : "  failed") );
                printf(    "(3)    | D(w/ Z) - D(w/o Z) | / |D|  = %8.2e%s\n\n", result[2], (result[2] < tolulp ? "" : "  failed") );
            }
            
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_B    );
            TESTING_FREE_CPU( w1     );
            TESTING_FREE_CPU( w2     );
            TESTING_FREE_CPU( rwork  );
            TESTING_FREE_CPU( iwork  );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_S    );
            TESTING_FREE_PIN( h_work );
        }                         
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return 0;
}
