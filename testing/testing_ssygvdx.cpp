/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @generated from testing/testing_zhegvdx.cpp normal z -> s, Mon May  2 23:31:19 2016

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define REAL

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssygvdx
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    /* Constants */
    const float c_zero    = MAGMA_S_ZERO;
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;
    
    /* Local variables */
    real_Double_t   gpu_time;
    float *h_A, *h_R, *h_B, *h_S, *h_Z, *h_work, aux_work[1];
    float *w1, *w2, abstol;
    float result[2] = {0};
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, n2, info, lwork, liwork, lda;
    #ifdef COMPLEX
    float *rwork, aux_rwork[1];
    magma_int_t lrwork;
    #endif
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    float tol    = opts.tolerance * lapackf77_slamch("E");
    float tolulp = opts.tolerance * lapackf77_slamch("P");
    
    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;
    
    #ifdef REAL
    if (opts.version == 2 || opts.version == 3) {
        printf("%% magma_ssygvr and magma_ssygvx are not available for real precisions (single, float).\n");
        return status;
    }
    #endif
    
    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% itype = %d, jobz = %s, range = %s, uplo = %s, fraction = %6.4f, ngpu = %d\n",
           int(opts.itype), lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           opts.fraction, int(abs_ngpu) );

    if (opts.itype == 1) {
        printf("%%   N     M   GPU Time (sec)   |AZ-BZD|   |D - D_magma|\n");
    }                                                   
    else if (opts.itype == 2) {                      
        printf("%%   N     M   GPU Time (sec)   |ABZ-ZD|   |D - D_magma|\n");
    }                                                   
    else if (opts.itype == 3) {                      
        printf("%%   N     M   GPU Time (sec)   |BAZ-ZD|   |D - D_magma|\n");
    }                                     
        printf("%%======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda = N;
            n2  = lda*N;

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
            abstol = 0;  // auto in ssygvr
            MAGMA_UNUSED( abstol );  // unused in [sd] precisions
            
            // query for workspace sizes
            if ( opts.version == 1 ) {
                magma_ssygvdx( opts.itype, opts.jobz, range, opts.uplo,
                               N, NULL, lda, NULL, lda,    // A, B
                               vl, vu, il, iu, &m1, NULL,  // w
                               aux_work,  -1,
                               #ifdef COMPLEX
                               aux_rwork, -1,
                               #endif
                               aux_iwork, -1,
                               &info );
            }
            else if ( opts.version == 2 ) {
                #ifdef COMPLEX
                magma_ssygvr( opts.itype, opts.jobz, range, opts.uplo,
                              N, NULL, lda, NULL, lda,  // A, B
                              vl, vu, il, iu, abstol,
                              &m1, NULL,                // w
                              NULL, lda, NULL,          // Z, isuppz
                              aux_work,  -1,
                              #ifdef COMPLEX
                              aux_rwork, -1,
                              #endif
                              aux_iwork, -1,
                              &info );
                #endif
            }
            else if ( opts.version == 3 ) {
                #ifdef COMPLEX
                magma_ssygvx( opts.itype, opts.jobz, range, opts.uplo,
                              N, NULL, lda, NULL, lda,  // A, B
                              vl, vu, il, iu, abstol,
                              &m1, NULL,         // w
                              NULL, lda,         // Z
                              aux_work,  -1,
                              #ifdef COMPLEX
                              aux_rwork,
                              #endif
                              aux_iwork,
                              NULL,              // ifail
                              &info );
                // ssyevx doesn't query rwork, iwork; set them for consistency
                aux_rwork[0] = float(7*N);
                aux_iwork[0] = float(5*N);
                #endif
            }
            lwork  = (magma_int_t) MAGMA_S_REAL( aux_work[0] );
            #ifdef COMPLEX
            lrwork = (magma_int_t) aux_rwork[0];
            #endif
            liwork = aux_iwork[0];
            
            TESTING_MALLOC_CPU( h_A,    float, n2     );
            TESTING_MALLOC_CPU( h_B,    float, n2     );
            TESTING_MALLOC_CPU( w1,     float,             N      );
            TESTING_MALLOC_CPU( w2,     float,             N      );
            TESTING_MALLOC_CPU( iwork,  magma_int_t,        liwork );
            
            TESTING_MALLOC_PIN( h_R,    float, n2     );
            TESTING_MALLOC_PIN( h_S,    float, n2     );
            TESTING_MALLOC_PIN( h_work, float, max( lwork, N*N ));  // check needs N*N
            #ifdef COMPLEX
            TESTING_MALLOC_PIN( rwork, float, lrwork );
            #endif
            
            if (opts.version == 2) {
                TESTING_MALLOC_CPU( h_Z,    float, N*lda      );
                TESTING_MALLOC_CPU( isuppz, magma_int_t,        2*max(1,N) );
            }
            if (opts.version == 3) {
                TESTING_MALLOC_CPU( h_Z,    float, N*lda      );
                TESTING_MALLOC_CPU( ifail,  magma_int_t,        N          );
            }
            
            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            lapackf77_slarnv( &ione, ISEED, &n2, h_B );
            magma_smake_hpd( N, h_B, lda );
            magma_smake_symmetric( N, h_A, lda );

            lapackf77_slacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            lapackf77_slacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */                
            gpu_time = magma_wtime();
            if (opts.version == 1) {
                if (opts.ngpu == 1) {
                    magma_ssygvdx( opts.itype, opts.jobz, range, opts.uplo,
                                   N, h_R, lda, h_S, lda, vl, vu, il, iu, &m1, w1,
                                   h_work, lwork,
                                   #ifdef COMPLEX
                                   rwork, lrwork,
                                   #endif
                                   iwork, liwork,
                                   &info );
                }
                else {
                    magma_ssygvdx_m( abs_ngpu, opts.itype, opts.jobz, range, opts.uplo,
                                     N, h_R, lda, h_S, lda, vl, vu, il, iu, &m1, w1,
                                     h_work, lwork,
                                     #ifdef COMPLEX
                                     rwork, lrwork,
                                     #endif
                                     iwork, liwork,
                                     &info );
                }
            }
            else if (opts.version == 2) {
                #ifdef COMPLEX
                magma_ssygvr( opts.itype, opts.jobz, range, opts.uplo,
                              N, h_R, lda, h_S, lda, vl, vu, il, iu, abstol, &m1, w1,
                              h_Z, lda, isuppz,
                              h_work, lwork,
                              #ifdef COMPLEX
                              rwork, lrwork,
                              #endif
                              iwork, liwork,
                              &info );
                lapackf77_slacpy( "Full", &N, &N, h_Z, &lda, h_R, &lda );
                #endif
            }
            else if (opts.version == 3) {
                #ifdef COMPLEX
                magma_ssygvx( opts.itype, opts.jobz, range, opts.uplo,
                              N, h_R, lda, h_S, lda, vl, vu, il, iu, abstol, &m1, w1,
                              h_Z, lda,
                              h_work, lwork,
                              #ifdef COMPLEX
                              rwork, /*lrwork,*/
                              #endif
                              iwork, /*liwork,*/
                              ifail,
                              &info );
                lapackf77_slacpy( "Full", &N, &N, h_Z, &lda, h_R, &lda );
                #endif
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_ssygvdx returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvdx routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A| |Z| N )  (itype = 1)
                          | A B Z - Z D | / ( |A| |Z| N )  (itype = 2)
                          | B A Z - Z D | / ( |A| |Z| N )  (itype = 3)
                   (2)    | D(with V, magma) - D(w/o V, lapack) | / | D |
                   =================================================================== */
                #ifdef REAL
                float *rwork = h_work + N*N;
                #endif
                
                if ( opts.jobz != MagmaNoVec ) {
                    result[0] = 1.;
                    result[0] /= safe_lapackf77_slansy("1", lapack_uplo_const(opts.uplo), &N, h_A, &lda, rwork);
                    result[0] /= lapackf77_slange("1", &N, &m1, h_R, &lda, rwork);
                    
                    if (opts.itype == 1) {
                        blasf77_ssymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                        for (int i=0; i < m1; ++i)
                            blasf77_sscal(&N, &w1[i], &h_R[i*N], &ione);
                        blasf77_ssymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_neg_one, h_B, &lda, h_R, &lda, &c_one, h_work, &N);
                        result[0] *= lapackf77_slange("1", &N, &m1, h_work, &N, rwork)/N;
                    }
                    else if (opts.itype == 2) {
                        blasf77_ssymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_B, &lda, h_R, &lda, &c_zero, h_work, &N);
                        for (int i=0; i < m1; ++i)
                            blasf77_sscal(&N, &w1[i], &h_R[i*N], &ione);
                        blasf77_ssymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                        result[0] *= lapackf77_slange("1", &N, &m1, h_R, &lda, rwork)/N;
                    }
                    else if (opts.itype == 3) {
                        blasf77_ssymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                        for (int i=0; i < m1; ++i)
                            blasf77_sscal(&N, &w1[i], &h_R[i*N], &ione);
                        blasf77_ssymm("L", lapack_uplo_const(opts.uplo), &N, &m1, &c_one, h_B, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                        result[0] *= lapackf77_slange("1", &N, &m1, h_R, &lda, rwork)/N;
                    }
                }
                
                lapackf77_slacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                lapackf77_slacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
                
                lapackf77_ssygvd( &opts.itype, "N", lapack_uplo_const(opts.uplo), &N,
                                  h_R, &lda, h_S, &lda, w2,
                                  h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork, &lrwork,
                                  #endif
                                  iwork, &liwork,
                                  &info );
                if (info != 0) {
                    printf("lapackf77_ssygvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                float maxw=0, diff=0;
                for (int j=0; j < m1; j++) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[1] = diff / (m1*maxw);
            }
            
            /* =====================================================================
               Print execution time
               =================================================================== */
            printf("%5d %5d   %9.4f     ",
                   (int) N, (int) m1, gpu_time);
            if ( opts.check ) {
                bool okay = (result[1] < tolulp);
                if ( opts.jobz != MagmaNoVec ) {
                    okay = okay && (result[0] < tol);
                    printf("   %8.2e", result[0] );
                }
                else {
                    printf("     ---   ");
                }
                printf("        %8.2e  %s\n", result[1], (okay ? "ok" : "failed"));
                status += ! okay;
            }
            else {
                printf("     ---\n");
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( w1  );
            TESTING_FREE_CPU( w2  );
            TESTING_FREE_CPU( iwork );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_S    );
            TESTING_FREE_PIN( h_work );
            #ifdef COMPLEX
            TESTING_FREE_PIN( rwork );
            #endif
            
            if ( opts.version == 2 ) {
                TESTING_FREE_CPU( h_Z    );
                TESTING_FREE_CPU( isuppz );
            }
            if ( opts.version == 3 ) {
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
