/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

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

#define PRECISION_d


#include "testings.h"

#define absv(v1) ((v1)>0? (v1): -(v1))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dsygvd
*/
int main( int argc, char** argv)
{
    TESTING_INIT_MGPU();

    double *h_A, *h_Ainit, *h_B, *h_Binit, *h_work;
    #if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    #endif
    double *w1, *w2, result;
    magma_int_t *iwork;
    double mgpu_time, gpu_time, cpu_time;

    /* Matrix size */
    magma_int_t N=0, n2;

    magma_int_t info;
    magma_int_t ione = 1;

    double c_zero    = MAGMA_D_ZERO;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    magma_int_t ISEED[4] = {0,0,0,1};

    magma_timestr_t start, end;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");

    char jobz = opts.jobz;
    int checkres = opts.check;

    char uplo = opts.uplo;
    magma_int_t itype = opts.itype;

    if ( checkres && jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        jobz = MagmaVec;
    }

    printf("using: nrgpu = %d, itype = %d, jobz = %c, uplo = %c, checkres = %d\n",
           (int) opts.ngpu, (int) itype, jobz, uplo, (int) checkres);

    printf("  N     M   nr GPU     MGPU Time(s) \n");
    printf("====================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            n2     = N*N;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            magma_int_t lwork = 2*N + N*N;
            magma_int_t lrwork = 1 + 5*N +2*N*N;
            // MKL's dsygvd has a bug for small N - it looks like what is returned by a 
            // query (consistent with LAPACK's number above) is different from a the memory
            // requirement ckeck (that returns info -11). The lwork increase below is needed
            // to pass this check.  
            if (N<32)
                lwork = 34*32;
            #else
            magma_int_t lwork  = 1 + 6*N + 2*N*N;
            #endif
            magma_int_t liwork = 3 + 5*N;

            TESTING_MALLOC_PIN( h_A,    double, n2    );
            TESTING_MALLOC_PIN( h_B,    double, n2    );
            TESTING_MALLOC_PIN( h_work, double, lwork );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_MALLOC_PIN( rwork, double, lrwork );
            #endif

            TESTING_MALLOC_CPU( w1,    double, N );
            TESTING_MALLOC_CPU( w2,    double, N );
            TESTING_MALLOC_CPU( iwork, magma_int_t, liwork );

            printf("  N     CPU Time(s)    GPU Time(s)   MGPU Time(s) \n");
            printf("==================================================\n");

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlarnv( &ione, ISEED, &n2, h_B );
            magma_dmake_hpd( N, h_B, N );
            magma_dmake_symmetric( N, h_A, N );

            if((opts.warmup)||( checkres )){
                TESTING_MALLOC_CPU( h_Ainit, double, n2 );
                TESTING_MALLOC_CPU( h_Binit, double, n2 );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_Ainit, &N );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_Binit, &N );
            }

            if(opts.warmup){

                // ==================================================================
                // Warmup using MAGMA.
                // ==================================================================
                magma_dsygvd_m( opts.ngpu, itype, jobz, uplo,
                                N, h_A, N, h_B, N, w1,
                                h_work, lwork,
                                #if defined(PRECISION_z) || defined(PRECISION_c)
                                rwork, lrwork,
                                #endif
                                iwork, liwork,
                                &info);
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_Ainit, &N, h_A, &N );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_Binit, &N, h_B, &N );
            }

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================

            start = get_current_time();
            magma_dsygvd_m( opts.ngpu, itype, jobz, uplo,
                            N, h_A, N, h_B, N, w1,
                            h_work, lwork,
                            #if defined(PRECISION_z) || defined(PRECISION_c)
                            rwork, lrwork,
                            #endif
                            iwork, liwork,
                            &info);
            end = get_current_time();

            if(info != 0)
                printf("magma_dsygvd_m returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));

            mgpu_time = GetTimerValue(start,end)/1000.;

            if ( checkres ) {
                /* =====================================================================
                 Check the results following the LAPACK's [zc]hegvd routine.
                 A x = lambda B x is solved
                 and the following 3 tests computed:
                 (1)    | A Z - B Z D | / ( |A||Z| N )  (itype = 1)
                 | A B Z - Z D | / ( |A||Z| N )  (itype = 2)
                 | B A Z - Z D | / ( |A||Z| N )  (itype = 3)
                 =================================================================== */

                #if defined(PRECISION_d) || defined(PRECISION_s)
                double *rwork = h_work + N*N;
                #endif

                result = 1.;
                result /= lapackf77_dlansy("1",&uplo, &N, h_Ainit, &N, rwork);
                result /= lapackf77_dlange("1",&N , &N, h_A, &N, rwork);

                if (itype == 1){
                    blasf77_dsymm("L", &uplo, &N, &N, &c_one, h_Ainit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_dscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_dsymm("L", &uplo, &N, &N, &c_neg_one, h_Binit, &N, h_A, &N, &c_one, h_work, &N);
                    result *= lapackf77_dlange("1", &N, &N, h_work, &N, rwork)/N;
                }
                else if (itype == 2){
                    blasf77_dsymm("L", &uplo, &N, &N, &c_one, h_Binit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_dscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_dsymm("L", &uplo, &N, &N, &c_one, h_Ainit, &N, h_work, &N, &c_neg_one, h_A, &N);
                    result *= lapackf77_dlange("1", &N, &N, h_A, &N, rwork)/N;
                }
                else if (itype == 3){
                    blasf77_dsymm("L", &uplo, &N, &N, &c_one, h_Ainit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_dscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_dsymm("L", &uplo, &N, &N, &c_one, h_Binit, &N, h_work, &N, &c_neg_one, h_A, &N);
                    result *= lapackf77_dlange("1", &N, &N, h_A, &N, rwork)/N;
                }

                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_Ainit, &N, h_A, &N );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_Binit, &N, h_B, &N );

                /* ====================================================================
                 Performs operation using MAGMA
                 =================================================================== */
                start = get_current_time();
                magma_dsygvd(itype, jobz, uplo,
                             N, h_A, N, h_B, N, w2,
                             h_work, lwork,
                             #if defined(PRECISION_z) || defined(PRECISION_c)
                             rwork, lrwork,
                             #endif
                             iwork, liwork,
                             &info);
                end = get_current_time();

                if(info != 0)
                    printf("magma_dsygvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));

                gpu_time = GetTimerValue(start,end)/1000.;

                /* =====================================================================
                 Performs operation using LAPACK
                 =================================================================== */
                start = get_current_time();
                lapackf77_dsygvd(&itype, &jobz, &uplo,
                                 &N, h_Ainit, &N, h_Binit, &N, w2,
                                 h_work, &lwork,
                                 #if defined(PRECISION_z) || defined(PRECISION_c)
                                 rwork, &lrwork,
                                 #endif
                                 iwork, &liwork,
                                 &info);
                end = get_current_time();
                if (info != 0)
                    printf("lapackf77_dsygvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));

                cpu_time = GetTimerValue(start,end)/1000.;

                double temp1 = 0;
                double temp2 = 0;
                for(int j=0; j<N; j++){
                    temp1 = max(temp1, absv(w1[j]));
                    temp1 = max(temp1, absv(w2[j]));
                    temp2 = max(temp2, absv(w1[j]-w2[j]));
                }
                double result2 = temp2 / (((double)N)*temp1);

                /* =====================================================================
                 Print execution time
                 =================================================================== */
                printf("%5d     %6.2f         %6.2f         %6.2f\n",
                       (int) N, cpu_time, gpu_time, mgpu_time);
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if(itype==1)
                    printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %8.2e%s\n", result, (result < tol ? "" : "  failed") );
                else if(itype==2)
                    printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %8.2e%s\n", result, (result < tol ? "" : "  failed") );
                else if(itype==3)
                    printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %8.2e%s\n", result, (result < tol ? "" : "  failed") );

                printf(    "(3)    | D(MGPU)-D(LAPACK) |/ |D|    = %8.2e%s\n\n", result2, (result2 < tolulp ? "" : "  failed") );
            }
            else {
                printf("%5d     ------         ------         %6.2f\n",
                       (int) N, mgpu_time);
            }

            /* Memory clean up */
            TESTING_FREE_PIN( h_A    );
            TESTING_FREE_PIN( h_B    );
            TESTING_FREE_PIN( h_work );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_FREE_PIN( rwork  );
            #endif
            
            TESTING_FREE_CPU( w1    );
            TESTING_FREE_CPU( w2    );
            TESTING_FREE_CPU( iwork );

            if((opts.warmup)||( checkres )){
                TESTING_FREE_CPU( h_Ainit );
                TESTING_FREE_CPU( h_Binit );
            }
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    /* Shutdown */
    TESTING_FINALIZE_MGPU();
}
