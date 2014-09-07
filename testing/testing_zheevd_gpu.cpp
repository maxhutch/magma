/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

    @author Raffaele Solca
    @author Azzam Haidar
    @author Stan Tomov

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


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zheevd_gpu
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *d_R, *h_work, aux_work[1];
    double *rwork, *w1, *w2, result[3], eps, aux_rwork[1];
    magma_int_t *iwork, aux_iwork[1];
    magma_int_t N, n2, info, lwork, lrwork, liwork, lda, ldda;
    magma_int_t izero    = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    eps = lapackf77_dlamch( "E" );
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");
    
    if ( opts.check && opts.jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVec;
    }
    
    printf("using: jobz = %s, uplo = %s\n",
           lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo));

    printf("    N   CPU Time (sec)   GPU Time (sec)\n");
    printf("=======================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2   = N*N;
            lda  = N;
            ldda = ((N + 31)/32)*32;
            
            // query for workspace sizes
            magma_zheevd_gpu( opts.jobz, opts.uplo,
                              N, NULL, ldda, NULL,
                              NULL, lda,
                              aux_work,  -1,
                              aux_rwork, -1,
                              aux_iwork, -1,
                              &info );
            lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
            lrwork = (magma_int_t) aux_rwork[0];
            liwork = aux_iwork[0];
            
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, N*lda  );
            TESTING_MALLOC_CPU( w1,     double,             N      );
            TESTING_MALLOC_CPU( w2,     double,             N      );
            TESTING_MALLOC_CPU( rwork,  double,             lrwork );
            TESTING_MALLOC_CPU( iwork,  magma_int_t,        liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, N*lda  );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork  );
            
            TESTING_MALLOC_DEV( d_R,    magmaDoubleComplex, N*ldda );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hermitian( N, h_A, N );
            
            magma_zsetmatrix( N, N, h_A, lda, d_R, ldda );
            
            /* warm up run */
            if ( opts.warmup ) {
                magma_zheevd_gpu( opts.jobz, opts.uplo,
                                  N, d_R, ldda, w1,
                                  h_R, lda,
                                  h_work, lwork,
                                  rwork, lrwork,
                                  iwork, liwork,
                                  &info );
                if (info != 0)
                    printf("magma_zheevd_gpu returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                magma_zsetmatrix( N, N, h_A, lda, d_R, ldda );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zheevd_gpu( opts.jobz, opts.uplo,
                              N, d_R, ldda, w1,
                              h_R, lda,
                              h_work, lwork,
                              rwork, lrwork,
                              iwork, liwork,
                              &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zheevd_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U' and the following 3 tests computed:
                   (1)    | A - U S U' | / ( |A| N )
                   (2)    | I - U'U | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |
                   =================================================================== */
                double temp1, temp2;
                
                // tau=NULL is unused since itype=1
                magma_zgetmatrix( N, N, d_R, ldda, h_R, lda );
                lapackf77_zhet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                  h_A, &lda,
                                  w1, w1,
                                  h_R, &lda,
                                  h_R, &lda,
                                  NULL, h_work, rwork, &result[0]);
                
                magma_zsetmatrix( N, N, h_A, lda, d_R, ldda );
                magma_zheevd_gpu( MagmaNoVec, opts.uplo,
                                  N, d_R, ldda, w2,
                                  h_R, lda,
                                  h_work, lwork,
                                  rwork, lrwork,
                                  iwork, liwork,
                                  &info);
                if (info != 0)
                    printf("magma_zheevd_gpu returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                temp1 = temp2 = 0;
                for( int j=0; j<N; j++ ) {
                    temp1 = max(temp1, fabs(w1[j]));
                    temp1 = max(temp1, fabs(w2[j]));
                    temp2 = max(temp2, fabs(w1[j]-w2[j]));
                }
                result[2] = temp2 / (((double)N)*temp1);
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zheevd(lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                 &N, h_A, &lda, w2,
                                 h_work, &lwork,
                                 rwork, &lrwork,
                                 iwork, &liwork,
                                 &info);
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_zheevd returned error %d: %s.\n",
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
                printf("Testing the factorization A = U S U' for correctness:\n");
                printf("(1)    | A - U S U' | / (|A| N)     = %8.2e   %s\n",   result[0]*eps, (result[0]*eps < tol ? "ok" : "failed") );
                printf("(2)    | I -   U'U  | /  N          = %8.2e   %s\n",   result[1]*eps, (result[1]*eps < tol ? "ok" : "failed") );
                printf("(3)    | S(w/ U) - S(w/o U) | / |S| = %8.2e   %s\n\n", result[2]    , (result[2]  < tolulp ? "ok" : "failed") );
                status += ! (result[0]*eps < tol && result[1]*eps < tol && result[2] < tolulp);
            }

            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( w1     );
            TESTING_FREE_CPU( w2     );
            TESTING_FREE_CPU( rwork  );
            TESTING_FREE_CPU( iwork  );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            
            TESTING_FREE_DEV( d_R );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
