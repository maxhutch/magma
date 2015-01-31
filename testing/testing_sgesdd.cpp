/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zgesdd.cpp normal z -> s, Fri Jan 30 19:00:26 2015
       @author Mark Gates

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

#define REAL

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgesdd (SVD with Divide & Conquer)
      Please keep code in testing_sgesdd.cpp and testing_sgesvd.cpp similar.
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    float *h_A, *h_R, *U, *VT, *h_work;
    float dummy[1];
    float *S1, *S2;
    #ifdef COMPLEX
    magma_int_t lrwork=0;
    float *rwork;
    #endif
    magma_int_t *iwork;
    
    magma_int_t M, N, N_U, M_VT, lda, ldu, ldv, n2, min_mn, max_mn, info, nb, lwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_vec_t jobz;
    magma_int_t status = 0;

    MAGMA_UNUSED( max_mn );  // used only in real
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    jobz = opts.jobu;
    
    magma_vec_t jobs[] = { MagmaNoVec, MagmaSomeVec, MagmaOverwriteVec, MagmaAllVec };
    
    if ( opts.check && ! opts.all && (jobz == MagmaNoVec)) {
        printf( "NOTE: some checks require that singular vectors are computed;\n"
                "      set jobz (option -U[NASO]) to be S, O, or A.\n\n" );
    }
    printf("jobz     M     N  CPU time (sec)  GPU time (sec)  |S1-S2|/.  |A-USV'|/. |I-UU'|/M  |I-VV'|/N  S sorted\n");
    printf("======================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int ijobz = 0; ijobz < 4; ++ijobz ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            if ( opts.all ) {
                jobz = jobs[ ijobz ];
            }
            else if ( ijobz > 0 ) {
                // if not testing all, run only once, when ijobz = 0,
                // but jobz come from opts (above loops).
                continue;
            }
            
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            max_mn = max(M, N);
            N_U  = (jobz == MagmaAllVec ? M : min_mn);
            M_VT = (jobz == MagmaAllVec ? N : min_mn);
            lda = M;
            ldu = M;
            ldv = M_VT;
            n2 = lda*N;
            nb = magma_get_sgesvd_nb(N);
            
            // x and y abbreviations used in sgesdd and dgesdd documentation
            magma_int_t x = max(M,N);
            magma_int_t y = min(M,N);
            #ifdef COMPLEX
            bool tall = (x >= int(y*17/9.));  // true if tall (m >> n) or wide (n >> m)
            #else
            bool tall = (x >= int(y*11/6.));  // true if tall (m >> n) or wide (n >> m)
            #endif
            
            // query or use formula for workspace size
            switch( opts.svd_work ) {
                case 0: {  // query for workspace size
                    lwork = -1;
                    magma_sgesdd( jobz, M, N,
                                  NULL, lda, NULL, NULL, ldu, NULL, ldv, dummy, lwork,
                                  #ifdef COMPLEX
                                  NULL,
                                  #endif
                                  NULL, &info );
                    lwork = (int) MAGMA_S_REAL( dummy[0] );
                    break;
                }
                
                case 1:    // minimum
                case 2:    // optimal
                case 3: {  // optimal (for gesdd, 2 & 3 are same; for gesvd, they differ)
                    // formulas from sgesdd and dgesdd documentation
                    bool sml = (opts.svd_work == 1);  // 1 is small workspace, 2,3 are large workspace
                    
                    #ifdef COMPLEX  // ----------------------------------------
                        if (jobz == MagmaNoVec) {
                            if (tall)    { lwork = 2*y + (2*y)*nb; }
                            else         { lwork = 2*y + (x+y)*nb; }
                        }
                        if (jobz == MagmaOverwriteVec) {
                            if (tall)    {
                                if (sml) { lwork = 2*y*y     + 2*y + (2*y)*nb; }
                                else     { lwork = y*y + x*y + 2*y + (2*y)*nb; }  // not big deal
                            }
                            else         {
                                //if (sml) { lwork = 2*y + max( (x+y)*nb, y*y + y    ); }
                                //else     { lwork = 2*y + max( (x+y)*nb, x*y + y*nb ); }
                                // LAPACK 3.4.2 over-estimates workspaces. For compatability, use these:
                                if (sml) { lwork = 2*y + max( (x+y)*nb, y*y + x ); }
                                else     { lwork = 2*y +      (x+y)*nb + x*y; }
                            }
                        }
                        if (jobz == MagmaSomeVec) {
                            if (tall)    { lwork = y*y + 2*y + (2*y)*nb; }
                            else         { lwork =       2*y + (x+y)*nb; }
                        }
                        if (jobz == MagmaAllVec) {
                            if (tall) {
                                if (sml) { lwork = y*y + 2*y + max( (2*y)*nb, x    ); }
                                else     { lwork = y*y + 2*y + max( (2*y)*nb, x*nb ); }
                            }
                            else         { lwork =       2*y +      (x+y)*nb; }
                        }
                    #else // REAL ----------------------------------------
                        if (jobz == MagmaNoVec) {
                            if (tall)    { lwork =       3*y + max( (2*y)*nb, 7*y ); }
                            else         { lwork =       3*y + max( (x+y)*nb, 7*y ); }
                        }
                        if (jobz == MagmaOverwriteVec) {
                            if (tall)    {
                                if (sml) { lwork = y*y + 3*y +      max( (2*y)*nb, 4*y*y + 4*y ); }
                                else     { lwork = y*y + 3*y + max( max( (2*y)*nb, 4*y*y + 4*y ), y*y + y*nb ); }
                            }
                            else         {
                                if (sml) { lwork =       3*y + max( (x+y)*nb, 4*y*y + 4*y ); }
                                else     { lwork =       3*y + max( (x+y)*nb, 3*y*y + 4*y + x*y ); }  // extra space not too important?
                            }
                        }
                        if (jobz == MagmaSomeVec) {
                            if (tall)    { lwork = y*y + 3*y + max( (2*y)*nb, 3*y*y + 4*y ); }
                            else         { lwork =       3*y + max( (x+y)*nb, 3*y*y + 4*y ); }
                        }
                        if (jobz == MagmaAllVec) {
                            if (tall) {
                                if (sml) { lwork = y*y + max( 3*y + max( (2*y)*nb, 3*y*y + 4*y ), y + x    ); }
                                else     { lwork = y*y + max( 3*y + max( (2*y)*nb, 3*y*y + 4*y ), y + x*nb ); }
                                // LAPACK 3.4.2 over-estimates workspaces. For compatability, use these:
                                //if (sml) { lwork = y*y +      3*y + max( (2*y)*nb,      3*y*y + 3*y + x ); }
                                //else     { lwork = y*y + max( 3*y + max( (2*y)*nb, max( 3*y*y + 3*y + x, 3*y*y + 4*y )), y + x*nb ); }
                            }
                            else         { lwork =            3*y + max( (x+y)*nb, 3*y*y + 4*y ); }
                        }
                    #endif
                    break;
                }
                
                default: {
                    fprintf( stderr, "Error: unknown option svd_work %d\n", (int) opts.svd_work );
                    exit(1);
                    break;
                }
            }
            
            TESTING_MALLOC_CPU( h_A,   float, lda*N );
            TESTING_MALLOC_CPU( VT,    float, ldv*N );   // N x N (jobz=A) or min(M,N) x N
            TESTING_MALLOC_CPU( U,     float, ldu*N_U ); // M x M (jobz=A) or M x min(M,N)
            TESTING_MALLOC_CPU( S1,    float, min_mn );
            TESTING_MALLOC_CPU( S2,    float, min_mn );
            TESTING_MALLOC_CPU( iwork, magma_int_t, 8*min_mn );
            TESTING_MALLOC_PIN( h_R,    float, lda*N );
            TESTING_MALLOC_PIN( h_work, float, lwork );
            
            #ifdef COMPLEX
                if (jobz == MagmaNoVec) {
                    // requires  5*min_mn, but MKL (11.1) seems to have a bug
                    // requiring 7*min_mn in some cases (e.g., jobz=N, m=100, n=170)
                    lrwork = 7*min_mn;
                }
                else if (tall) {
                    lrwork = 5*min_mn*min_mn + 5*min_mn;
                }
                else {
                    lrwork = max( 5*min_mn*min_mn + 5*min_mn,
                                  2*max_mn*min_mn + 2*min_mn*min_mn + min_mn );
                }
                TESTING_MALLOC_CPU( rwork, float, lrwork );
            #endif
            
            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            lapackf77_slacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_sgesdd( jobz, M, N,
                          h_R, lda, S1, U, ldu, VT, ldv, h_work, lwork,
                          #ifdef COMPLEX
                          rwork,
                          #endif
                          iwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_sgesdd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));

            float eps = lapackf77_slamch( "E" );
            float result[5] = { -1/eps, -1/eps, -1/eps, -1/eps, -1/eps };
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvbd routine.
                   A is factored as A = U diag(S) VT and the following 4 tests computed:
                   (1)    | A - U diag(S) VT | / ( |A| max(M,N) )
                   (2)    | I - U'U | / ( M )
                   (3)    | I - VT VT' | / ( N )
                   (4)    S contains MNMIN nonnegative values in decreasing order.
                          (Return 0 if true, 1/ULP if false.)
                   =================================================================== */
                magma_int_t izero = 0;
                
                // get size and location of U and V^T depending on jobz
                // U2=NULL and VT2=NULL if they were not computed (e.g., jobz=N)
                float *U2  = NULL;
                float *VT2 = NULL;
                if ( jobz == MagmaSomeVec || jobz == MagmaAllVec ) {
                    U2  = U;
                    VT2 = VT;
                }
                else if ( jobz == MagmaOverwriteVec ) {
                    if ( M >= N ) {
                        U2  = h_R;
                        ldu = lda;
                        VT2 = VT;
                    }
                    else {
                        U2  = U;
                        VT2 = h_R;
                        ldv = lda;
                    }
                }
                
                // sbdt01 needs M+N
                // sort01 prefers N*(N+1) to check U; M*(M+1) to check V
                magma_int_t lwork_err = M+N;
                if ( U2 != NULL ) {
                    lwork_err = max( lwork_err, N_U*(N_U+1) );
                }
                if ( VT2 != NULL ) {
                    lwork_err = max( lwork_err, M_VT*(M_VT+1) );
                }
                float *h_work_err;
                TESTING_MALLOC_CPU( h_work_err, float, lwork_err );
                
                // sbdt01 and sort01 need max(M,N), depending
                float *rwork_err;
                TESTING_MALLOC_CPU( rwork_err, float, max(M,N) );
                
                if ( U2 != NULL && VT2 != NULL ) {
                    // since KD=0 (3rd arg), E is not referenced so pass NULL (9th arg)
                    lapackf77_sbdt01(&M, &N, &izero, h_A, &lda,
                                     U2, &ldu, S1, NULL, VT2, &ldv,
                                     h_work_err,
                                     #ifdef COMPLEX
                                     rwork_err,
                                     #endif
                                     &result[0]);
                }
                if ( U2 != NULL ) {
                    lapackf77_sort01("Columns", &M,  &N_U, U2,  &ldu, h_work_err, &lwork_err,
                                     #ifdef COMPLEX
                                     rwork_err,
                                     #endif
                                     &result[1]);
                }
                if ( VT2 != NULL ) {
                    lapackf77_sort01("Rows",    &M_VT, &N, VT2, &ldv, h_work_err, &lwork_err,
                                     #ifdef COMPLEX
                                     rwork_err,
                                     #endif
                                     &result[2]);
                }
                
                result[3] = 0.;
                for(int j=0; j < min_mn-1; j++){
                    if ( S1[j] < S1[j+1] )
                        result[3] = 1.;
                    if ( S1[j] < 0. )
                        result[3] = 1.;
                }
                if (min_mn > 1 && S1[min_mn-1] < 0.)
                    result[3] = 1.;
                
                result[0] *= eps;
                result[1] *= eps;
                result[2] *= eps;
                
                TESTING_FREE_CPU( h_work_err );
                TESTING_FREE_CPU( rwork_err );
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_sgesdd( lapack_vec_const(jobz), &M, &N,
                                  h_A, &lda, S2, U, &ldu, VT, &ldv, h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  iwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_sgesdd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                float work[1], c_neg_one = -1;
                magma_int_t one = 1;
                
                blasf77_saxpy(&min_mn, &c_neg_one, S1, &one, S2, &one);
                result[4]  = lapackf77_slange("f", &min_mn, &one, S2, &min_mn, work);
                result[4] /= lapackf77_slange("f", &min_mn, &one, S1, &min_mn, work);
                
                printf("   %c %5d %5d  %7.2f         %7.2f         %8.2e",
                       lapack_vec_const(jobz)[0],
                       (int) M, (int) N, cpu_time, gpu_time, result[4] );
            }
            else {
                printf("   %c %5d %5d    ---           %7.2f           ---   ",
                       lapack_vec_const(jobz)[0],
                       (int) M, (int) N, gpu_time );
            }
            if ( opts.check ) {
                if ( result[0] < 0. ) { printf("     ---   "); } else { printf("  %#9.3g", result[0]); }
                if ( result[1] < 0. ) { printf("     ---   "); } else { printf("  %#9.3g", result[1]); }
                if ( result[2] < 0. ) { printf("     ---   "); } else { printf("  %#9.3g", result[2]); }
                int success = (result[0] < tol) && (result[1] < tol) && (result[2] < tol) && (result[3] == 0.) && (result[4] < tol);
                printf("   %3s   %s\n", (result[3] == 0. ? "yes" : "no"), (success ? "ok" : "failed"));
                status += ! success;
            }
            else {
                printf("\n");
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( VT  );
            TESTING_FREE_CPU( U   );
            TESTING_FREE_CPU( S1  );
            TESTING_FREE_CPU( S2  );
            #ifdef COMPLEX
            TESTING_FREE_CPU( rwork );
            #endif
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            fflush( stdout );
        }}
        if ( opts.all || opts.niter > 1 ) {
            printf("\n");
        }
    }

    TESTING_FINALIZE();
    return status;
}
