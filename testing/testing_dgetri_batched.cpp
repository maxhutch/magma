/*
   -- MAGMA (version 1.6.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date January 2015

   @author Azzam Haidar

   @generated from testing_zgetri_batched.cpp normal z -> d, Fri Jan 30 19:00:26 2015
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double *h_A, *h_R;
    double *d_A, *d_invA;
    double **dA_array = NULL;
    double **dinvA_array = NULL;
    double **C_array = NULL;
    magma_int_t  **dipiv_array = NULL;
    magma_int_t *dinfo_array = NULL;

    magma_int_t     *ipiv;
    magma_int_t     *d_ipiv, *d_info;
    magma_int_t M, N, n2, lda, ldda, min_mn, info, info1, info2;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check; 

    magma_queue_t queue = magma_stream;
    magma_int_t batchCount = opts.batchcount ;
    magma_int_t columns;
    double error=0.0, rwork[1];
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t     status = 0;
    // need looser bound (3000*eps instead of 30*eps) for tests
    // TODO: should compute ||I - A*A^{-1}|| / (n*||A||*||A^{-1}||)
    opts.tolerance = max( 3000., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("batchCount      M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)    ||PA-LU||/(||A||*N    tolerance )\n");
    printf("====================================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
    
      for( int iter = 0; iter < opts.niter; ++iter ) {
            
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda   = ((M+31)/32)*32;
            //gflops = (FLOPS_DGETRF( M, N ) + FLOPS_DGETRI( min(M,N) ))/ 1e9 * batchCount; // This is the correct flops but since this getri_batched is based on 2 trsm = getrs and to know the real flops I am using the getrs one
            gflops = (FLOPS_DGETRF( M, N ) + FLOPS_DGETRS( min(M,N), min(M,N) ))/ 1e9 * batchCount;

            TESTING_MALLOC_CPU(  ipiv, magma_int_t,     min_mn * batchCount);
            TESTING_MALLOC_CPU(  h_A,  double, n2     );
            TESTING_MALLOC_PIN(  h_R,  double, n2     );
            TESTING_MALLOC_DEV(  d_A,  double, ldda*N * batchCount);
            TESTING_MALLOC_DEV(  d_invA,  double, ldda*N * batchCount);
            TESTING_MALLOC_DEV(  d_ipiv,  magma_int_t, min_mn * batchCount);
            TESTING_MALLOC_DEV(  d_info,  magma_int_t, batchCount);


            magma_malloc((void**)&dA_array, batchCount * sizeof(*dA_array));
            magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
            magma_malloc((void**)&dinfo_array, batchCount * sizeof(magma_int_t));
            magma_malloc((void**)&C_array, batchCount * sizeof(*C_array));
            magma_malloc((void**)&dipiv_array, batchCount * sizeof(*dipiv_array));

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            columns = N * batchCount;
            lapackf77_dlacpy( MagmaUpperLowerStr, &M, &columns, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( M, columns, h_R, lda, d_A, ldda );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            dset_pointer(dA_array, d_A, ldda, 0, 0, ldda * N, batchCount, queue);
            dset_pointer(dinvA_array, d_invA, ldda, 0, 0, ldda * N, batchCount, queue);
            set_ipointer(dipiv_array, d_ipiv, 1, 0, 0, min(M,N), batchCount, queue);

            gpu_time = magma_sync_wtime(0);
            info1 = magma_dgetrf_batched( M, N, dA_array, ldda, dipiv_array, dinfo_array, batchCount, queue);
            info2 = magma_dgetri_outofplace_batched( min(M,N), dA_array, ldda, dipiv_array, dinvA_array, ldda, dinfo_array, batchCount, queue);
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gflops / gpu_time;


            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_int_t *cpu_info = (magma_int_t*) malloc(batchCount*sizeof(magma_int_t));
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
            for(int i=0; i<batchCount; i++)
            {
                if(cpu_info[i] != 0 ){
                    printf("magma_dgetrf_batched matrix %d returned error %d\n", i, (int)cpu_info[i] );
                }
            }
            if (info1 != 0) printf("magma_dgetrf_batched returned argument error %d: %s.\n", (int) info1, magma_strerror( info1 ));
            if (info2 != 0) printf("magma_dgetri_batched returned argument error %d: %s.\n", (int) info2, magma_strerror( info2 ));
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                // query for workspace size
                double *work;
                double tmp;
                magma_int_t lwork = -1;
                lapackf77_dgetri( &N, NULL, &lda, NULL, &tmp, &lwork, &info );
                if (info != 0)
                    printf("lapackf77_dgetri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                lwork = magma_int_t( MAGMA_D_REAL( tmp ));
                TESTING_MALLOC_CPU( work,  double, lwork  );
                lapackf77_dlacpy( MagmaUpperLowerStr, &M, &columns, h_R, &lda, h_A, &lda );
                cpu_time = magma_wtime();
                for(int i=0; i<batchCount; i++)
                {
                    lapackf77_dgetrf(&M, &N, h_A + i * lda*N, &lda, ipiv + i * min_mn, &info);
                    if (info != 0)
                        printf("lapackf77_dgetrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                    lapackf77_dgetri(&N, h_A + i * lda*N, &lda, ipiv + i * min_mn, work, &lwork, &info );
                    if (info != 0)
                        printf("lapackf77_dgetri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10d %6d %6d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) batchCount, (int) M, (int) N, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000. );
            }
            else {
                printf("%10d %6d %6d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) batchCount, (int) M, (int) N, gpu_perf, gpu_time*1000. );
            }

            double err = 0.0;
            if ( opts.check ) {
                magma_getvector( min_mn * batchCount, sizeof(magma_int_t), d_ipiv, 1, ipiv, 1 );
                magma_dgetmatrix( min(M,N), N*batchCount, d_invA, ldda, h_R, lda );
                int stop=0;
                n2     = lda*N;
                for(int i=0; i < batchCount; i++)
                {
                    for(int k=0; k < min_mn; k++){
                        if(ipiv[i*min_mn+k] < 1 || ipiv[i*min_mn+k] > M )
                        {
                            printf("error for matrix %d ipiv @ %d = %d\n", (int) i, (int) k, (int) ipiv[i*min_mn+k]);
                            stop=1;
                        }
                    }
                    if(stop==1){
                        err=-1.0;
                        break;
                    }
                    error = lapackf77_dlange( "f", &N, &N, h_A+ i * lda*N, &lda, rwork );
                    blasf77_daxpy( &n2, &c_neg_one, h_A+ i * lda*N, &ione, h_R+ i * lda*N, &ione );
                    error = lapackf77_dlange( "f", &N, &N, h_R+ i * lda*N, &lda, rwork ) / (N*error);
                    if ( isnan(error) || isinf(error) ) {
                        err = error;
                        break;
                    }
                    err = max(fabs(error), err);
                }
                printf("   %18.2e   %10.2e   %s\n", err, tol, (err < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("     ---  \n");
            }

            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_invA );
            TESTING_FREE_DEV( d_ipiv );
            TESTING_FREE_DEV( d_info );
            TESTING_FREE_DEV( dipiv_array );
            TESTING_FREE_DEV( dA_array );
            TESTING_FREE_DEV( dinfo_array );
            TESTING_FREE_DEV( C_array );
            free(cpu_info);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    TESTING_FINALIZE();
    return status;
}
