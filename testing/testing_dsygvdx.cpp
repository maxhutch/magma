/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    @author Raffaele Solca
    @author Azzam Haidar

    @generated d Tue Dec 17 13:18:57 2013

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

#define PRECISION_d

#define absv(v1) ((v1)>0? (v1): -(v1))
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dsygvdx
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time /*cpu_time*/;
    double *h_A, *h_R, *h_B, *h_S, *h_work;
    double *w1, *w2, vl=0, vu=0;
    double result[2] = {0};
    magma_int_t *iwork;
    magma_int_t N, n2, info, il, iu, m1, m2, nb, lwork, liwork;
    double c_zero    = MAGMA_D_ZERO;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
#if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    magma_int_t lrwork;
#endif
    //double d_one         =  1.;
    //double d_ten         = 10.;
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
    
    printf("    N     M   GPU Time (sec)\n");
    printf("============================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            n2     = N*N;
            nb     = magma_get_dsytrd_nb(N);
#if defined(PRECISION_z) || defined(PRECISION_c)
            lwork  = 2*N*nb + N*N;
            lrwork = 1 + 5*N +2*N*N;
#else
            lwork  = 1 + 6*N*nb + 2* N*N;
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

            TESTING_MALLOC_CPU( h_A,    double, n2     );
            TESTING_MALLOC_CPU( h_B,    double, n2     );
            TESTING_MALLOC_CPU( w1,     double,             N      );
            TESTING_MALLOC_CPU( w2,     double,             N      );
            TESTING_MALLOC_CPU( iwork,  magma_int_t,        liwork );
            
            TESTING_MALLOC_PIN( h_R,    double, n2     );
            TESTING_MALLOC_PIN( h_S,    double, n2     );
            TESTING_MALLOC_PIN( h_work, double, lwork  );
#if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_MALLOC_PIN( rwork, double, lrwork);
#endif
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlarnv( &ione, ISEED, &n2, h_B );
            magma_dmake_hpd( N, h_B, N );
            magma_dmake_symmetric( N, h_A, N );

            // ==================================================================
            // Warmup using MAGMA
            // ==================================================================
            if(opts.warmup){
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
                
                magma_dsygvdx( opts.itype, opts.jobz, 'I', opts.uplo,
                               N, h_R, N, h_S, N, vl, vu, il, iu, &m1, w1,
                               h_work, lwork,
#if defined(PRECISION_z) || defined(PRECISION_c)
                               rwork, lrwork,
#endif      
                               iwork, liwork,
                               &info );
                if (info != 0)
                    printf("magma_dsygvdx returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );

            gpu_time = magma_wtime();
            magma_dsygvdx( opts.itype, opts.jobz, 'I', opts.uplo,
                           N, h_R, N, h_S, N, vl, vu, il, iu, &m1, w1,
                           h_work, lwork,
#if defined(PRECISION_z) || defined(PRECISION_c)
                           rwork, lrwork,
#endif
                           iwork, liwork,
                           &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_dsygvdx returned error %d: %s.\n",
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
                result[0] /= lapackf77_dlansy("1", &opts.uplo, &N, h_A, &N, rwork);
                result[0] /= lapackf77_dlange("1", &N, &m1, h_R, &N, rwork);
                
                if (opts.itype == 1) {
                    blasf77_dsymm("L", &opts.uplo, &N, &m1, &c_one, h_A, &N, h_R, &N, &c_zero, h_work, &N);
                    for(int i=0; i < m1; ++i)
                        blasf77_dscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", &opts.uplo, &N, &m1, &c_neg_one, h_B, &N, h_R, &N, &c_one, h_work, &N);
                    result[0] *= lapackf77_dlange("1", &N, &m1, h_work, &N, rwork)/N;
                }
                else if (opts.itype == 2) {
                    blasf77_dsymm("L", &opts.uplo, &N, &m1, &c_one, h_B, &N, h_R, &N, &c_zero, h_work, &N);
                    for(int i=0; i < m1; ++i)
                        blasf77_dscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", &opts.uplo, &N, &m1, &c_one, h_A, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_dlange("1", &N, &m1, h_R, &N, rwork)/N;
                }
                else if (opts.itype == 3) {
                    blasf77_dsymm("L", &opts.uplo, &N, &m1, &c_one, h_A, &N, h_R, &N, &c_zero, h_work, &N);
                    for(int i=0; i < m1; ++i)
                        blasf77_dscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", &opts.uplo, &N, &m1, &c_one, h_B, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_dlange("1", &N, &m1, h_R, &N, rwork)/N;
                }
                
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
                
                magma_dsygvdx( opts.itype, 'N', 'I', opts.uplo,
                               N, h_R, N, h_S, N, vl, vu, il, iu, &m2, w2,
                               h_work, lwork,
#if defined(PRECISION_z) || defined(PRECISION_c)
                               rwork, lrwork,
#endif
                               iwork, liwork,
                               &info );
                if (info != 0)
                    printf("magma_dsygvdx returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                temp1 = temp2 = 0;
                for(int j=0; j < m2; j++) {
                    temp1 = max(temp1, absv(w1[j]));
                    temp1 = max(temp1, absv(w2[j]));
                    temp2 = max(temp2, absv(w1[j]-w2[j]));
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
                if (opts.itype==1)
                    printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %8.2e%s\n", result[0], (result[0] < tol ? "" : "  failed"));
                else if (opts.itype==2)
                    printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %8.2e%s\n", result[0], (result[0] < tol ? "" : "  failed"));
                else if (opts.itype==3)
                    printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %8.2e%s\n", result[0], (result[0] < tol ? "" : "  failed"));
                printf(    "(2)    | D(w/ Z) - D(w/o Z) | / |D|  = %8.2e%s\n\n", result[1], (result[1] < tolulp ? "" : "  failed"));
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
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
