/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> c d s
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

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesvd (SVD with QR iteration)
      Please keep code in testing_zgesdd.cpp and testing_zgesvd.cpp similar.
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *U, *VT, *h_work;
    magmaDoubleComplex dummy[1];
    double *S1, *S2;
    #ifdef COMPLEX
    double *rwork;
    #endif
    magma_int_t M, N, N_U, M_VT, lda, ldu, ldv, n2, min_mn, info, nb, lwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_vec_t jobu, jobvt;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    jobu  = opts.jobu;
    jobvt = opts.jobvt;
    
    magma_vec_t jobs[] = { MagmaNoVec, MagmaSomeVec, MagmaOverwriteVec, MagmaAllVec };
    
    if ( opts.check && ! opts.all && (jobu == MagmaNoVec || jobvt == MagmaNoVec)) {
        printf( "NOTE: some checks require that singular vectors are computed;\n"
                "      set both jobu (option -U[NASO]) and jobvt (option -V[NASO]) to be S, O, or A.\n\n" );
    }
    printf("jobu jobv     M     N  CPU time (sec)  GPU time (sec)  |S1-S2|/.  |A-USV'|/. |I-UU'|/M  |I-VV'|/N  S sorted\n");
    printf("===========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int ijobu = 0; ijobu < 4; ++ijobu ) {
        for( int ijobv = 0; ijobv < 4; ++ijobv ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            if ( opts.all ) {
                jobu  = jobs[ ijobu ];
                jobvt = jobs[ ijobv ];
            }
            else if ( ijobu > 0 || ijobv > 0 ) {
                // if not testing all, run only once, when ijobu = ijobv = 0,
                // but jobu and jobv come from opts (above loops).
                continue;
            }
            if ( jobu == MagmaOverwriteVec && jobvt == MagmaOverwriteVec ) {
                // illegal combination; skip
                continue;
            }
            
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            N_U  = (jobu  == MagmaAllVec ? M : min_mn);
            M_VT = (jobvt == MagmaAllVec ? N : min_mn);
            lda = M;
            ldu = M;
            ldv = M_VT;
            n2 = lda*N;
            nb = magma_get_zgesvd_nb(N);
            
            // query or use formula for workspace size 
            switch( opts.svd_work ) {
                case 0: {  // query for workspace size
                    lwork = -1;
                    magma_zgesvd( jobu, jobvt, M, N,
                                  h_R, lda, S1, U, ldu, VT, ldv, dummy, lwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &info );
                    lwork = (int) MAGMA_Z_REAL( dummy[0] );
                    break;
                }
                
                // these are not tight bounds:
                // if (m >> n), it may need only (2*N)*nb instead of (M+N)*nb.
                // if (n >> m), it may need only (2*M)*nb instead of (M+N)*nb.
                #ifdef COMPLEX
                case 1: lwork = (M+N)*nb + 2*min_mn;                   break;  // minimum
                case 2: lwork = (M+N)*nb + 2*min_mn +   min_mn*min_mn; break;  // optimal for some paths
                case 3: lwork = (M+N)*nb + 2*min_mn + 2*min_mn*min_mn; break;  // optimal for all paths
                #else
                case 1: lwork = (M+N)*nb + 3*min_mn;                   break;  // minimum
                case 2: lwork = (M+N)*nb + 3*min_mn +   min_mn*min_mn; break;  // optimal for some paths
                case 3: lwork = (M+N)*nb + 3*min_mn + 2*min_mn*min_mn; break;  // optimal for all paths
                #endif
                
                default: {
                    fprintf( stderr, "Error: unknown option svd_work %d\n", (int) opts.svd_work );
                    exit(1);
                    break;
                }
            }
            
            TESTING_MALLOC_CPU( h_A,   magmaDoubleComplex, lda*N );
            TESTING_MALLOC_CPU( VT,    magmaDoubleComplex, ldv*N );   // N x N (jobvt=A) or min(M,N) x N
            TESTING_MALLOC_CPU( U,     magmaDoubleComplex, ldu*N_U ); // M x M (jobu=A)  or M x min(M,N)
            TESTING_MALLOC_CPU( S1,    double, min_mn );
            TESTING_MALLOC_CPU( S2,    double, min_mn );
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, lda*N );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            
            #ifdef COMPLEX
            TESTING_MALLOC_CPU( rwork, double, 5*min_mn );
            #endif
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgesvd( jobu, jobvt, M, N,
                          h_R, lda, S1, U, ldu, VT, ldv, h_work, lwork,
                          #ifdef COMPLEX
                          rwork,
                          #endif
                          &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zgesvd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            double eps = lapackf77_dlamch( "E" );
            double result[5] = { -1/eps, -1/eps, -1/eps, -1/eps, -1/eps };
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
                
                // get size and location of U and V^T depending on jobu and jobvt
                // U2=NULL and VT2=NULL if they were not computed (e.g., jobu=N)
                magmaDoubleComplex *U2  = NULL;
                magmaDoubleComplex *VT2 = NULL;
                if ( jobu == MagmaSomeVec || jobu == MagmaAllVec ) {
                    U2 = U;
                }
                else if ( jobu == MagmaOverwriteVec ) {
                    U2 = h_R;
                    ldu = lda;
                }
                if ( jobvt == MagmaSomeVec || jobvt == MagmaAllVec ) {
                    VT2 = VT;
                }
                else if ( jobvt == MagmaOverwriteVec ) {
                    VT2 = h_R;
                    ldv = lda;
                }
                
                // zbdt01 needs M+N
                // zunt01 prefers N*(N+1) to check U; M*(M+1) to check V
                magma_int_t lwork_err = M+N;
                if ( U2 != NULL ) {
                    lwork_err = max( lwork_err, N_U*(N_U+1) );
                }
                if ( VT2 != NULL ) {
                    lwork_err = max( lwork_err, M_VT*(M_VT+1) );
                }
                magmaDoubleComplex *h_work_err;
                TESTING_MALLOC_CPU( h_work_err, magmaDoubleComplex, lwork_err );
                
                // zbdt01 and zunt01 need max(M,N), depending
                double *rwork_err;
                TESTING_MALLOC_CPU( rwork_err, double, max(M,N) );                
                
                if ( U2 != NULL && VT2 != NULL ) {
                    // since KD=0 (3rd arg), E is not referenced so pass NULL (9th arg)
                    lapackf77_zbdt01(&M, &N, &izero, h_A, &lda,
                                     U2, &ldu, S1, NULL, VT2, &ldv,
                                     h_work_err,
                                     #ifdef COMPLEX
                                     rwork_err,
                                     #endif
                                     &result[0]);
                }
                if ( U2 != NULL ) {
                    lapackf77_zunt01("Columns", &M,  &N_U, U2,  &ldu, h_work_err, &lwork_err,
                                     #ifdef COMPLEX
                                     rwork_err,
                                     #endif
                                     &result[1]);
                }
                if ( VT2 != NULL ) {
                    lapackf77_zunt01("Rows",    &M_VT, &N, VT2, &ldv, h_work_err, &lwork_err,
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
                lapackf77_zgesvd( lapack_vec_const(jobu), lapack_vec_const(jobvt), &M, &N,
                                  h_A, &lda, S2, U, &ldu, VT, &ldv, h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &info);
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_zgesvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                double work[1], c_neg_one = -1;
                magma_int_t one = 1;
                
                blasf77_daxpy(&min_mn, &c_neg_one, S1, &one, S2, &one);
                result[4]  = lapackf77_dlange("f", &min_mn, &one, S2, &min_mn, work);
                result[4] /= lapackf77_dlange("f", &min_mn, &one, S1, &min_mn, work);
                
                printf("   %c    %c %5d %5d  %7.2f         %7.2f         %8.2e",
                       lapack_vec_const(jobu)[0], lapack_vec_const(jobvt)[0],
                       (int) M, (int) N, cpu_time, gpu_time, result[4] );
            }
            else {
                printf("   %c    %c %5d %5d    ---           %7.2f           ---   ",
                       lapack_vec_const(jobu)[0], lapack_vec_const(jobvt)[0],
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
        }}}
        if ( opts.all || opts.niter > 1 ) {
            printf("\n");
        }
    }

    TESTING_FINALIZE();
    return status;
}
