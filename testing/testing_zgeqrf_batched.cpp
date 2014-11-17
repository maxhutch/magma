/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "common_magma.h"



extern "C" magma_int_t
magma_zgeqrf_batched_v1(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magmaDoubleComplex **dA_array,
        magma_int_t lda, magmaDoubleComplex **dtau_array,
        magmaDoubleComplex *dT, 
        magmaDoubleComplex **dT_array,  magma_int_t ldt,
        magmaDoubleComplex *dR, 
        magmaDoubleComplex **dR_array, magma_int_t ldr,
        double *dwork, magma_int_t *info, magma_int_t batchCount);

extern "C" magma_int_t
magma_zgeqrf_batched_v2(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magmaDoubleComplex **dA_array,
        magma_int_t lda, magmaDoubleComplex **dtau_array,
        magmaDoubleComplex *dT, 
        magmaDoubleComplex **dT_array,  magma_int_t ldt,
        magmaDoubleComplex *dR, 
        magmaDoubleComplex **dR_array, magma_int_t ldr,
        double *dwork, magma_int_t *info, magma_int_t batchCount);



extern "C" magma_int_t
magma_zgeqrf_batched_v4(
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex *dA,
    magmaDoubleComplex **A_array,
    magma_int_t lda, 
    magmaDoubleComplex **tau_array,
    magmaDoubleComplex *dT,
    magmaDoubleComplex **dT_array, magma_int_t ldt, 
    magmaDoubleComplex *dR,
    magmaDoubleComplex **dR_array, magma_int_t ldr,
    double *dnorm, magmaDoubleComplex *dwork, 
    magma_int_t *info, magma_int_t batchCount);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double           error, work[1];

    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R, *h_C, *tau, *h_work, tmp[1];
    magmaDoubleComplex *d_A_magma, *d_A_cublas, *d_T, *d_R, *dtau_magma, *dtau_cublas, *dwork;
    double *dnorm;
    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dR_array = NULL;
    magmaDoubleComplex **dtau_array = NULL; 
    magmaDoubleComplex **dT_array = NULL; 
    

    magma_int_t M, N, lda, ldda, lwork, n2, info, min_mn, ldt, ldr, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_int_t batchCount = 1;
    magma_int_t batchSize;
    magma_int_t column;

    #define NNB 32

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    batchCount = opts.batchcount ;

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    printf("BatchCount     M    N     MAGMA GFlop/s (ms)   CUBLAS GFlop/s (ms)   CPU GFlop/s (ms)   ||R||_F/||A||_F  ||R_T||\n");
    printf("=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            //   ldda   = ((M+31)/32)*32;
            ldda = M;
            nb = NNB;

            magma_int_t NN = N*N * batchCount;
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRT( M, N )) / 1e9 * batchCount;

            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( tau,   magmaDoubleComplex, min_mn * batchCount );
            TESTING_MALLOC_CPU( h_A,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_R,   magmaDoubleComplex, nb * N * batchCount);
        
            TESTING_MALLOC_PIN( h_C,   magmaDoubleComplex, n2     );
        
            TESTING_MALLOC_DEV( d_A_magma,   magmaDoubleComplex, n2     );
            //TESTING_MALLOC_DEV( d_A_cublas,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_DEV( d_T,   magmaDoubleComplex, nb*nb*batchCount    );
            TESTING_MALLOC_DEV( d_R,   magmaDoubleComplex, nb * N * batchCount    );
            TESTING_MALLOC_DEV( dtau_magma,  magmaDoubleComplex, min_mn * batchCount);
            TESTING_MALLOC_DEV( dtau_cublas,  magmaDoubleComplex, min_mn * batchCount);

            magma_malloc((void**)&dA_array, batchCount * sizeof(*dA_array));
            magma_malloc((void**)&dR_array, batchCount * sizeof(*dR_array));
            magma_malloc((void**)&dtau_array, batchCount * sizeof(*dtau_array));
            magma_malloc((void**)&dT_array, batchCount * sizeof(*dT_array));
        
            TESTING_MALLOC_DEV( dnorm,  double, min_mn * batchCount );
            TESTING_MALLOC_DEV( dwork,  magmaDoubleComplex, (2 * nb * min_mn +3*nb ) * batchCount );

            // todo replace with magma_zlaset
            cudaMemset(d_R, 0, N*nb*batchCount*sizeof(magmaDoubleComplex));
            cudaMemset(d_T, 0, nb*nb*batchCount*sizeof(magmaDoubleComplex));
        
            // to determine the size of lwork
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            lwork = max(lwork, N*N);        
           
            //printf("lwork, which is size of h_work = %d\n", lwork);
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork * batchCount);

            column = N * batchCount;
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &column, h_A, &lda, h_C, &lda );


            ldt = ldr = nb;
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */

            magma_zsetmatrix( M, column, h_C, lda,  d_A_magma, ldda );
            zset_pointer(dA_array, d_A_magma, 1, 0, 0, ldda*N, batchCount); 
            zset_pointer(dR_array, d_R, 1, 0, 0, nb*N, batchCount);
            zset_pointer(dT_array,  d_T, 1, 0, 0, nb*nb, batchCount);
            zset_pointer(dtau_array, dtau_magma, 1, 0, 0, min_mn, batchCount);
    
            magma_time = magma_sync_wtime(0);
    
            //magma_zgeqr2x2_gpu(&M, &N, d_A_magma, &ldda, dtau_magam, d_T, d_R, dwork, &info);
            //magma_zgeqrf_batched_v1(M, N, d_A_magma, dA_array, ldda, dtau_array, d_T, dT_array, ldt, d_R, dR_array, ldr, dwork, &info, batchCount);
            magma_zgeqrf_batched_v4(M, N, d_A_magma, dA_array, ldda, dtau_array, d_T, dT_array, ldt, d_R, dR_array, ldr, dnorm, dwork, &info, batchCount);

            magma_time = magma_sync_wtime(0) - magma_time;
            magma_perf = gflops / magma_time;
            if (info != 0)
                printf("magma_zgeqrf_batched returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using CUBLAS 
               =================================================================== */
#if 0
            /* This routine is only available from CUBLAS v6.5 */
            magma_zsetmatrix( M, column, h_C, lda,  d_A_cublas, ldda );
            zset_pointer(dA_array, d_A_cublas, 1, 0, 0, ldda*N, batchCount); 
            zset_pointer(dtau_array, dtau_cublas, 1, 0, 0, min_mn, batchCount);

            cublas_time = magma_sync_wtime(0);
    
            cublasZgeqrfBatched(opts.handle, M, N, dA_array, ldda, dtau_array, &info, batchCount);

            cublas_time = magma_sync_wtime(0) - cublas_time;
            cublas_perf = gflops / cublas_time;

            if (info != 0)
                printf("cublas_zgeqrf_batched returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
#endif
            /* =====================================================================
                   Performs operation using LAPACK
            =================================================================== */

            if ( opts.check ) {

                cpu_time = magma_wtime();
                #pragma unroll
                for(int i=0; i<batchCount; i++)
                {
                    lapackf77_zgeqrf(&M, &N, h_A + i*lda*N, &lda, tau + i*min_mn, h_work + i * lwork, &lwork, &info);
                    lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                                     &M, &N, h_A + i*lda*N, &lda, tau + i*min_mn, h_work + i * lwork, &N);
                }
                //magma_zgeqr2(&M, &N, h_A, &lda, tau, h_work, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                

                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_zgetmatrix( M, column, d_A_magma, ldda, h_C, lda);
                magma_zgetmatrix( nb, N*batchCount, d_R, nb, h_R, nb );
    
                /*
                // Restore the upper triangular part of A before the check
        int offset = 0;
                #pragma unroll
        for(int j=0; j<(N-1)/nb+1; j++)
                {

                    for(int i=0; i<batchCount; i++)
            {            
                        for(int col=0; col < nb; col++)
                        {
                            for(int row=0; row <= col; row++)
                {
                                h_C[ j*( nb + nb * lda) +  row + col*lda + i * lda * N] = h_R[ j * nb * nb +  row + col*nb + i * nb * N];
                                //printf("h_R=%f \t",  h_R[ j * nb * nb +  row + col*nb + i * nb * N]);
                }
            }
            }    
                }
                */
                
                error = lapackf77_zlange("M", &M, &column, h_A, &lda, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_C, &ione);
                error = lapackf77_zlange("M", &M, &column, h_C, &lda, work) / error;
                      
                double terr = 0.;
       

                if(batchCount == 1)
                {
                // Check if T is the same
                magma_zgetmatrix( nb, nb*batchCount, d_T, ldt, h_R, ldr );

                #pragma unroll
                for(int i=0; i<batchCount; i++)
                {
                for(int col=0; col < nb; col++)
                    for(int row=0; row <= col; row++)
                            {
                        terr += (  MAGMA_Z_ABS(h_work[row + col*N + i * lwork] - h_R[row + col*nb + i * nb * nb])*
                                   MAGMA_Z_ABS(h_work[row + col*N + i * lwork] - h_R[row + col*nb + i * nb * nb])  );
                                //printf("error of T is %e", terr);
                            }    
                }
                terr = magma_dsqrt(terr);
                }  
                printf("%5d       %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)     %8.2e     %8.2e   %s\n",
                       (int)batchCount, (int) M, (int) N, magma_perf, 1000.*magma_time, cublas_perf, 1000.*cublas_time,cpu_perf, 1000.*cpu_time, 
                       error, terr, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5d       %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)      ---   (  ---  )   ---  \n",
                       (int)batchCount, (int) M, (int) N, magma_perf, 1000.*magma_time, cublas_perf, 1000.*cublas_time);
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_R    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_C    );
        
            TESTING_FREE_DEV( d_A_magma   );
            TESTING_FREE_DEV( d_A_cublas  );
            TESTING_FREE_DEV( d_T   );
            TESTING_FREE_DEV( d_R   );
            TESTING_FREE_DEV( dtau_magma  );
            TESTING_FREE_DEV( dtau_cublas );
            TESTING_FREE_DEV( dnorm );
            TESTING_FREE_DEV( dwork );

            TESTING_FREE_DEV( dA_array   );
            TESTING_FREE_DEV( dR_array   );
            TESTING_FREE_DEV( dtau_array  );
            TESTING_FREE_DEV( dT_array );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );

    TESTING_FINALIZE();
    return status;
}
