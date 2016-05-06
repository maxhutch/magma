/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @generated from testing/testing_zheevdx_2stage.cpp normal z -> c, Mon May  2 23:31:19 2016

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
#include "magma_threadsetting.h"

#define COMPLEX

static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaFloatComplex *Q, magma_int_t LDQ, float eps);
static magma_int_t check_reduction(magma_uplo_t uplo, magma_int_t N, magma_int_t bw, magmaFloatComplex *A, float *D, magma_int_t LDA, magmaFloatComplex *Q, float eps );
static magma_int_t check_solution(magma_int_t N, float *E1, float *E2, float eps);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chegvdx
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t gpu_time;

    magmaFloatComplex *h_A, *h_R, *h_work;

    #ifdef COMPLEX
    float *rwork;
    magma_int_t lrwork;
    #endif

    float *w1, *w2;
    magma_int_t *iwork;
    magma_int_t N, n2, info, lda, lwork, liwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info_ortho     = 0;
    magma_int_t info_solution  = 0;
    magma_int_t info_reduction = 0;
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% jobz = %s, range = %s, uplo = %s, fraction = %6.4f, ngpu %d\n",
           lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           opts.fraction, int(abs_ngpu) );

    printf("%%   N     M  GPU Time (sec)   ||I-Q^H Q||/N   ||A-QDQ^H||/(||A||N)   |D-D_magma|/(|D| * N)\n");
    printf("%%=========================================================================================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
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

            magma_cheevdx_getworksize(N, threads, (opts.jobz == MagmaVec), 
                                     &lwork, 
                                     #ifdef COMPLEX
                                     &lrwork, 
                                     #endif
                                     &liwork);
            
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( h_A,   magmaFloatComplex, n2 );
            TESTING_MALLOC_CPU( w1,    float, N );
            TESTING_MALLOC_CPU( w2,    float, N );
            TESTING_MALLOC_CPU( iwork, magma_int_t, liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, n2    );
            TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lwork );
            #ifdef COMPLEX
            TESTING_MALLOC_PIN( rwork, float, lrwork );
            #endif

            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            magma_cmake_hermitian( N, h_A, lda );

            if (opts.warmup) {
                // ==================================================================
                // Warmup using MAGMA
                // ==================================================================
                lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                if (opts.ngpu == 1) {
                    //printf("calling cheevdx_2stage 1 GPU\n");
                    magma_cheevdx_2stage( opts.jobz, range, opts.uplo, N, 
                                          h_R, lda, 
                                          vl, vu, il, iu, 
                                          &m1, w1, 
                                          h_work, lwork, 
                                          #ifdef COMPLEX
                                          rwork, lrwork, 
                                          #endif
                                          iwork, liwork, 
                                          &info );
                } else {
                    //printf("calling cheevdx_2stage_m %d GPU\n", (int) opts.ngpu);
                    magma_cheevdx_2stage_m( abs_ngpu, opts.jobz, range, opts.uplo, N, 
                                            h_R, lda, 
                                            vl, vu, il, iu, 
                                            &m1, w1, 
                                            h_work, lwork, 
                                            #ifdef COMPLEX
                                            rwork, lrwork, 
                                            #endif
                                            iwork, liwork, 
                                            &info );
                }
            }

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================
            lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                //printf("calling cheevdx_2stage 1 GPU\n");
                magma_cheevdx_2stage( opts.jobz, range, opts.uplo, N, 
                                      h_R, lda, 
                                      vl, vu, il, iu, 
                                      &m1, w1, 
                                      h_work, lwork, 
                                      #ifdef COMPLEX
                                      rwork, lrwork, 
                                      #endif
                                      iwork, liwork, 
                                      &info );
            } else {
                //printf("calling cheevdx_2stage_m %d GPU\n", (int) opts.ngpu);
                magma_cheevdx_2stage_m( abs_ngpu, opts.jobz, range, opts.uplo, N, 
                                        h_R, lda, 
                                        vl, vu, il, iu, 
                                        &m1, w1, 
                                        h_work, lwork, 
                                        #ifdef COMPLEX
                                        rwork, lrwork, 
                                        #endif
                                        iwork, liwork, 
                                        &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_cheevdx_2stage returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            printf("%5d %5d  %7.2f      ",
                   (int) N, (int) m1, gpu_time );

            if ( opts.check ) {
                info_solution  = 0;
                info_ortho     = 0;
                info_reduction = 0;
                //float eps   = lapackf77_slamch("E")*lapackf77_slamch("B");
                float eps   = lapackf77_slamch("E");
              
                /* Check the orthogonality, reduction and the eigen solutions */
                if (opts.jobz == MagmaVec) {
                    info_ortho = check_orthogonality(N, N, h_R, lda, eps);
                    info_reduction = check_reduction(opts.uplo, N, 1, h_A, w1, lda, h_R, eps);
                } else {
                    printf("         ---                ---  ");
                }
                lapackf77_cheevd("N", "L", &N, 
                                h_A, &lda, w2, 
                                h_work, &lwork, 
                                #ifdef COMPLEX
                                rwork, &lrwork, 
                                #endif
                                iwork, &liwork, 
                                &info);
                info_solution = check_solution(N, w2, w1, eps);
                
                bool okay = (info_solution == 0) && (info_ortho == 0) && (info_reduction == 0);
                status += ! okay;
                printf("  %s", (okay ? "ok" : "failed"));
            }
            printf("\n");

            TESTING_FREE_CPU( h_A   );
            TESTING_FREE_CPU( w1    );
            TESTING_FREE_CPU( w2    );
            TESTING_FREE_CPU( iwork );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            #ifdef COMPLEX
            TESTING_FREE_PIN( rwork  );
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



/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaFloatComplex *Q, magma_int_t LDQ, float eps)
{
    float d_one     =  1.0;
    float d_neg_one = -1.0;
    magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    float  normQ, result;
    magma_int_t     info_ortho;
    magma_int_t     minMN = min(M, N);
    float *work;
    TESTING_MALLOC_CPU( work, float, minMN );

    /* Build the idendity matrix */
    magmaFloatComplex *Id;
    TESTING_MALLOC_CPU( Id, magmaFloatComplex, minMN*minMN );
    lapackf77_claset("A", &minMN, &minMN, &c_zero, &c_one, Id, &minMN);

    /* Perform Id - Q^H Q */
    if (M >= N)
        blasf77_cherk("U", "C", &N, &M, &d_one, Q, &LDQ, &d_neg_one, Id, &N);
    else
        blasf77_cherk("U", "N", &M, &N, &d_one, Q, &LDQ, &d_neg_one, Id, &M);

    normQ = safe_lapackf77_clanhe("I", "U", &minMN, Id, &minMN, work);

    result = normQ / (minMN * eps);
    printf( "      %8.2e", normQ / minMN );

    // TODO: use opts.tolerance instead of hard coding 60
    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        info_ortho = 1;
    }
    else {
        info_ortho = 0;
    }
    TESTING_FREE_CPU( work );
    TESTING_FREE_CPU( Id   );
    
    return info_ortho;
}


/*------------------------------------------------------------
 *  Check the reduction 
 */
static magma_int_t check_reduction(magma_uplo_t uplo, magma_int_t N, magma_int_t bw, magmaFloatComplex *A, float *D, magma_int_t LDA, magmaFloatComplex *Q, float eps )
{
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *TEMP, *Residual;
    float *work;
    float Anorm, Rnorm, result;
    magma_int_t info_reduction;
    magma_int_t i;
    magma_int_t ione=1;

    TESTING_MALLOC_CPU( TEMP,     magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( Residual, magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( work,     float, N );
    
    /* Compute TEMP =  Q * LAMBDA */
    lapackf77_clacpy("A", &N, &N, Q, &LDA, TEMP, &N);        
    for (i = 0; i < N; i++) {
        blasf77_csscal(&N, &D[i], &(TEMP[i*N]), &ione);
    }
    /* Compute Residual = A - Q * LAMBDA * Q^H */
    /* A is Hermitian but both upper and lower 
     * are assumed valable here for checking 
     * otherwise it need to be symetrized before 
     * checking.
     */ 
    lapackf77_clacpy("A", &N, &N, A, &LDA, Residual, &N);        
    blasf77_cgemm("N", "C", &N, &N, &N, &c_neg_one, TEMP, &N, Q, &LDA, &c_one, Residual,     &N);

    // since A has been generated by larnv and we did not symmetrize, 
    // so only the uplo portion of A should be equal to Q*LAMBDA*Q^H 
    // for that Rnorm use clanhe instead of clange
    Rnorm = safe_lapackf77_clanhe("1", lapack_uplo_const(uplo), &N, Residual, &N, work);
    Anorm = safe_lapackf77_clanhe("1", lapack_uplo_const(uplo), &N, A,        &LDA, work);

    result = Rnorm / ( Anorm * N * eps);
    printf("           %8.2e",  Rnorm / ( Anorm * N));

    // TODO: use opts.tolerance instead of hard coding 60
    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        info_reduction = 1;
    }
    else {
        info_reduction = 0;
    }

    TESTING_FREE_CPU( TEMP     );
    TESTING_FREE_CPU( Residual );
    TESTING_FREE_CPU( work     );

    return info_reduction;
}


/*------------------------------------------------------------
 *  Check the eigenvalues 
 */
static magma_int_t check_solution(magma_int_t N, float *E1, float *E2, float eps)
{
    magma_int_t   info_solution, i;
    float unfl   = lapackf77_slamch("Safe minimum");
    float resid;
    float maxtmp;
    float maxdif = fabs( fabs(E1[0]) - fabs(E2[0]) );
    float maxeig = max( fabs(E1[0]), fabs(E2[0]) );
    for (i = 1; i < N; i++) {
        resid   = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp  = max(fabs(E1[i]), fabs(E2[i]));

        /* Update */
        maxeig = max(maxtmp, maxeig);
        maxdif  = max(resid,  maxdif );
    }
    maxtmp = maxdif / max(unfl, eps*max(maxeig, maxdif));

    printf("              %8.2e", maxdif / (max(maxeig, maxdif)) );

    // TODO: use opts.tolerance instead of hard coding 100
    if ( isnan(maxtmp) || isinf(maxtmp) || (maxtmp > 100) ) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }
    return info_solution;
}
