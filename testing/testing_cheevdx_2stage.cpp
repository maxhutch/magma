/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @generated from testing_zheevdx_2stage.cpp normal z -> c, Fri Jan 30 19:00:26 2015

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
#include "magma_cbulge.h"
#include "magma_threadsetting.h"

#define PRECISION_c

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

    #if defined(PRECISION_z) || defined(PRECISION_c)
    float *rwork;
    magma_int_t lrwork;
    #endif

    /* Matrix size */
    float *w1, *w2;
    magma_int_t *iwork;
    magma_int_t N, n2, info, lwork, liwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};;
    magma_int_t info_ortho     = 0;
    magma_int_t info_solution  = 0;
    magma_int_t info_reduction = 0;
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;

    if ( opts.check && opts.jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVec;
    }

    printf("using: itype = %d, jobz = %s, range = %s, uplo = %s, check = %d, fraction = %6.4f\n",
           (int) opts.itype, lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           (int) opts.check, opts.fraction);

    printf("    N     M  GPU Time (sec)  ||I-Q'Q||/.  ||A-QDQ'||/.  ||D-D_magma||/.\n");
    printf("=======================================================================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2     = N*N;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lwork  = magma_cbulge_get_lq2(N, threads) + 2*N + N*N;
            lrwork = 1 + 5*N +2*N*N;
            #else
            lwork  = magma_cbulge_get_lq2(N, threads) + 1 + 6*N + 2*N*N;
            #endif
            liwork = 3 + 5*N;

            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( h_A,   magmaFloatComplex, n2 );
            TESTING_MALLOC_CPU( w1,    float, N );
            TESTING_MALLOC_CPU( w2,    float, N );
            TESTING_MALLOC_CPU( iwork, magma_int_t, liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, n2    );
            TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lwork );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_MALLOC_PIN( rwork, float, lrwork );
            #endif

            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            magma_cmake_hermitian( N, h_A, N );

            magma_int_t m1 = 0;
            float vl = 0;
            float vu = 0;
            magma_int_t il = 0;
            magma_int_t iu = 0;
            if (range == MagmaRangeI) {
                il = 1;
                iu = (int) (opts.fraction*N);
            }

            if (opts.warmup) {
                // ==================================================================
                // Warmup using MAGMA
                // ==================================================================
                lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                if (opts.ngpu == 1) {
                    //printf("calling cheevdx_2stage 1 GPU\n");
                    magma_cheevdx_2stage(opts.jobz, range, opts.uplo, N, 
                                    h_R, N, 
                                    vl, vu, il, iu, 
                                    &m1, w1, 
                                    h_work, lwork, 
                                    #if defined(PRECISION_z) || defined(PRECISION_c)
                                    rwork, lrwork, 
                                    #endif
                                    iwork, liwork, 
                                    &info);
                } else {
                    //printf("calling cheevdx_2stage_m %d GPU\n", (int) opts.ngpu);
                    magma_cheevdx_2stage_m(opts.ngpu, opts.jobz, range, opts.uplo, N, 
                                    h_R, N, 
                                    vl, vu, il, iu, 
                                    &m1, w1, 
                                    h_work, lwork, 
                                    #if defined(PRECISION_z) || defined(PRECISION_c)
                                    rwork, lrwork, 
                                    #endif
                                    iwork, liwork, 
                                    &info);
                }
            }


            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================
            lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                //printf("calling cheevdx_2stage 1 GPU\n");
                magma_cheevdx_2stage(opts.jobz, range, opts.uplo, N, 
                                h_R, N, 
                                vl, vu, il, iu, 
                                &m1, w1, 
                                h_work, lwork, 
                                #if defined(PRECISION_z) || defined(PRECISION_c)
                                rwork, lrwork, 
                                #endif
                                iwork, liwork, 
                                &info);
           
            } else {
                //printf("calling cheevdx_2stage_m %d GPU\n", (int) opts.ngpu);
                magma_cheevdx_2stage_m(opts.ngpu, opts.jobz, range, opts.uplo, N, 
                                h_R, N, 
                                vl, vu, il, iu, 
                                &m1, w1, 
                                h_work, lwork, 
                                #if defined(PRECISION_z) || defined(PRECISION_c)
                                rwork, lrwork, 
                                #endif
                                iwork, liwork, 
                                &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            
            printf("%5d %5d  %7.2f      ",
                   (int) N, (int) m1, gpu_time );

            if ( opts.check && opts.jobz != MagmaNoVec ) {
                float eps   = lapackf77_slamch("E");
                //printf("\n");
                //printf("------ TESTS FOR MAGMA CHEEVD ROUTINE -------  \n");
                //printf("        Size of the Matrix %d by %d\n", (int) N, (int) N);
                //printf("\n");
                //printf(" The matrix A is randomly generated for each test.\n");
                //printf("============\n");
                //printf(" The relative machine precision (eps) is %8.2e\n",eps);
                //printf(" Computational tests pass if scaled residuals are less than 60.\n");
              
                /* Check the orthogonality, reduction and the eigen solutions */
                if (opts.jobz == MagmaVec) {
                    info_ortho = check_orthogonality(N, N, h_R, N, eps);
                    info_reduction = check_reduction(opts.uplo, N, 1, h_A, w1, N, h_R, eps);
                }
                //printf("------ CALLING LAPACK CHEEVD TO COMPUTE only eigenvalue and verify elementswise -------  \n");
                lapackf77_cheevd("N", "L", &N, 
                                h_A, &N, w2, 
                                h_work, &lwork, 
                                #if defined(PRECISION_z) || defined(PRECISION_c)
                                rwork, &lrwork, 
                                #endif
                                iwork, &liwork, 
                                &info);
                info_solution = check_solution(N, w2, w1, eps);
              
                if ( (info_solution == 0) && (info_ortho == 0) && (info_reduction == 0) ) {
                    printf("  ok\n");
                    //printf("***************************************************\n");
                    //printf(" ---- TESTING CHEEVD ...................... PASSED !\n");
                    //printf("***************************************************\n");
                }
                else {
                    printf("  failed\n");
                    status += 1;
                    //printf("************************************************\n");
                    //printf(" - TESTING CHEEVD ... FAILED !\n");
                    //printf("************************************************\n");
                }
            }

            TESTING_FREE_CPU( h_A   );
            TESTING_FREE_CPU( w1    );
            TESTING_FREE_CPU( w2    );
            TESTING_FREE_CPU( iwork );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_FREE_PIN( rwork  );
            #endif
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



/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaFloatComplex *Q, magma_int_t LDQ, float eps)
{
    float  done  =  1.0;
    float  mdone = -1.0;
    magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    float  normQ, result;
    magma_int_t     info_ortho;
    magma_int_t     minMN = min(M, N);
    float *work;
    magma_smalloc_cpu( &work, minMN );

    /* Build the idendity matrix */
    magmaFloatComplex *Id;
    magma_cmalloc_cpu( &Id, minMN*minMN );
    lapackf77_claset("A", &minMN, &minMN, &c_zero, &c_one, Id, &minMN);

    /* Perform Id - Q'Q */
    if (M >= N)
        blasf77_cherk("U", "C", &N, &M, &done, Q, &LDQ, &mdone, Id, &N);
    else
        blasf77_cherk("U", "N", &M, &N, &done, Q, &LDQ, &mdone, Id, &M);

    normQ = lapackf77_clanhe("I", "U", &minMN, Id, &minMN, work);

    result = normQ / (minMN * eps);
    printf( "  %12.2e", result*eps );
    //printf(" ======================================================\n");
    //printf(" ||Id-Q'*Q||_oo / (minMN*eps)          : %15.3E \n",  result );
    //printf(" ======================================================\n");

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        //printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        //printf("-- Orthogonality is CORRECT ! \n");
        info_ortho=0;
    }
    magma_free_cpu(work);
    magma_free_cpu(Id);
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

    magma_cmalloc_cpu( &TEMP, N*N );
    magma_cmalloc_cpu( &Residual, N*N );
    magma_smalloc_cpu( &work, N );
    
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
    Rnorm = lapackf77_clanhe("1", lapack_uplo_const(uplo), &N, Residual, &N, work);
    Anorm = lapackf77_clanhe("1", lapack_uplo_const(uplo), &N, A,        &LDA, work);

    result = Rnorm / ( Anorm * N * eps);
    printf("  %12.2e", result );
    //if ( uplo == MagmaLower ) {
    //    printf(" ======================================================\n");
    //    printf(" ||A-Q*LAMBDA*Q'||_oo/(||A||_oo.N.eps) : %15.3E \n",  result );
    //    printf(" ======================================================\n");
    //} else { 
    //    printf(" ======================================================\n");
    //    printf(" ||A-Q'*LAMBDA*Q||_oo/(||A||_oo.N.eps) : %15.3E \n",  result );
    //    printf(" ======================================================\n");
    //}

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        //printf("-- Reduction is suspicious ! \n");
        info_reduction = 1;
    }
    else {
        //printf("-- Reduction is CORRECT ! \n");
        info_reduction = 0;
    }

    magma_free_cpu(TEMP);
    magma_free_cpu(Residual);
    magma_free_cpu(work);

    return info_reduction;
}


/*------------------------------------------------------------
 *  Check the eigenvalues 
 */
static magma_int_t check_solution(magma_int_t N, float *E1, float *E2, float eps)
{
    magma_int_t info_solution, i;
    float resid;
    float maxtmp;
    float maxel = fabs( fabs(E1[0]) - fabs(E2[0]) );
    float maxeig = max( fabs(E1[0]), fabs(E2[0]) );
    for (i = 1; i < N; i++) {
        resid   = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp  = max(fabs(E1[i]), fabs(E2[i]));

        /* Update */
        maxeig = max(maxtmp, maxeig);
        maxel  = max(resid,  maxel );
    }

    maxel = maxel / (maxeig * N * eps);
    printf("  %12.2e", maxel*eps );
    //printf(" ======================================================\n");
    //printf(" | D - eigcomputed | / (|D| * N * eps) : %15.3E \n",  maxel );
    //printf(" ======================================================\n");

    if ( isnan(maxel) || isinf(maxel) || (maxel > 100) ) {
        //printf("-- The eigenvalues are suspicious ! \n");
        info_solution = 1;
    }
    else {
        //printf("-- The eigenvalues are CORRECT ! \n");
        info_solution = 0;
    }
    return info_solution;
}

