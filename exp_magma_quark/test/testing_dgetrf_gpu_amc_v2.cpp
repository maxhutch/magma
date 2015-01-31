/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated d Thu Jun 13 11:49:58 2013
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
//#include <mkl_service.h>
// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "magma_amc.h"
//#include "magma_threadsetting.h"
#include "common_magma.h"

double get_LU_error(magma_int_t M, magma_int_t N,
                    double *A,  magma_int_t lda,
                    double *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    double alpha = MAGMA_D_ONE;
    double beta  = MAGMA_D_ZERO;
    double *L, *U;
    double work[1], matnorm, residual;
    

    TESTING_MALLOC_CPU( L, double, M*min_mn);
    TESTING_MALLOC_CPU( U, double, min_mn*N);
    memset( L, 0, M*min_mn*sizeof(double) );
    memset( U, 0, min_mn*N*sizeof(double) );

    lapackf77_dlaswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_dlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_dlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = MAGMA_D_MAKE( 1., 0. );
    
    matnorm = lapackf77_dlange("f", &M, &N, A, &lda, work);

    blasf77_dgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_D_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_dlange("f", &M, &N, LU, &lda, work);

    TESTING_FREE_CPU(L);
    TESTING_FREE_CPU(U);

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgetrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   cpu_perf, cpu_time;
    real_Double_t   gflops;
    real_Double_t   gpu_perf1, gpu_time1, gpu_perf2, gpu_time2, gpu_perf3, gpu_time3;

    double          error;
    double *h_A, *h_R;
    double *d_A;
    magma_int_t     *ipiv;
    magma_int_t M, N, n2, lda, ldda, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_int_t P=-1; /*Number of threads in the CPU part*/
    double d_cpu=-1;/*pourcentgae of the matrix to allocate in the cpu part*/
    magma_int_t Pr=-1;    /*Number of threads for the panel*/
    magma_int_t nb;
    magma_int_t lapack_numthreads = 1;
    double *WORK;
    magma_int_t WORK_LD, WORK_n;

    magma_opts opts;


    parse_opts( argc, argv, &opts );
    
    
    /*
    if (argc != 1){
        for(int i = 1; i<argc; i++){
            if (strcmp("-P", argv[i])==0)
            {
                P = atoi(argv[++i]);
            }
            else if (strcmp("-d", argv[i])==0)
            {
                dcpu = atoi(argv[++i]);
            }
            else if (strcmp("-r", argv[i])==0)
            {
                Pr = atoi(argv[++i]);
            }
        }
    }*/

    P =  opts.nthread;
    
    nb = opts.nb;
    
    //if(nb==0)
    // nb  = magma_get_dgetrf_nb(m) ;//magma dgetrf block size

    /*TODO: set these parameters in parse_opts*/
    Pr = opts.panel_nthread;

    d_cpu = 0.0;
    #if defined(CPU_PEAK) && defined(GPU_PEAK)
    d_cpu = magma_amc_recommanded_dcpu(opts.nthread, CPU_PEAK, 1, GPU_PEAK); //assume 1 GPU
    #endif
    if(opts.fraction_dcpu!=0){ /*Overwrite the one computed with the model*/
    d_cpu = opts.fraction_dcpu;
    }
    magma_assert(d_cpu > 0 && d_cpu<=1.0,
    "error: The cpu fraction is invalid. Ensure you use --fraction_dcpu with fraction_dcpu in [0.0, 1.0] or compile with both -DCPU_PEAK=<cpu peak performance> and -DGPU_PEAK=<gpu peak performance> set.\n");
    
    printf("Asynchronous recursif LU... nb:%d, nbcores:%d, dcpu:%f, panel_nbcores:%d\n", nb, P, d_cpu, Pr);
    printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   GPU_Async GFlop/s (sec)  GPU_Async_work GFlop/s (sec)     ||PA-LU||/(||A||*N)\n");
    printf("=========================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_DGETRF( M, N ) / 1e9;
            cpu_perf=gpu_perf1=gpu_perf2=gpu_perf3=0.0;

            TESTING_MALLOC_CPU(    ipiv, magma_int_t,     min_mn );
            TESTING_MALLOC_CPU(    h_A,  double, n2     );
            TESTING_MALLOC_PIN( h_R,  double, n2     );
            TESTING_MALLOC_DEV(  d_A,  double, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

            /*set default number of threads for lapack*/
            magma_setlapack_numthreads(P);
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_dgetrf(&M, &N, h_A, &lda, ipiv, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("\nlapackf77_dgetrf returned error %d: %s.\n     ",
                           (int) info, magma_strerror( info ));      
            }
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            
            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
            gpu_time1 = magma_wtime();
            magma_dgetrf_gpu( M, N, d_A, ldda, ipiv, &info);
            gpu_time1 = magma_wtime() - gpu_time1;
            gpu_perf1 = gflops / gpu_time1;
            if (info != 0)
                printf("\nmagma_dgetrf_gpu returned error %d: %s.\n     ",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA_ASYNC
               =================================================================== */
            /*Save BLAS number of threads*/
            /*
            lapack_numthreads = magma_getlapack_numthreads();
            if(lapack_numthreads !=1){ 
                //Temporary set the number of threads to 1.
                printf("MAGMA num threads set from %d to %d\n", lapack_numthreads, 1);
                magma_setlapack_numthreads(1);
            }*/

            /* Switch to the sequential version of BLAS*/
            magma_setlapack_numthreads(1);

            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
            magma_amc_init(P, d_cpu, Pr, nb);
            gpu_time2 = magma_wtime();
            magma_dgetrf_gpu_amc_v2( M, N, d_A, ldda, ipiv, &info);
            gpu_time2 = magma_wtime() - gpu_time2;
            gpu_perf2 = gflops / gpu_time2;
            magma_amc_finalize();
            if (info != 0)
                printf("\nmagma_amc_dgetrf_rec_gpu returned error %d: %s.\n     ",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA_ASYNC..._WORK (Workspace preallocated version)
               =================================================================== */

            /* Switch to sequential BLAS*/
            magma_setlapack_numthreads(1);
            /*TEST: Temporary use P*/
            //magma_setlapack_numthreads(P);
            //mkl_set_num_threads(P);
            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );

            /*Compute workspace dimension*/
            WORK_LD = M;
            WORK_n = (int) ceil(N*d_cpu);

            if(WORK_n<nb) WORK_n = nb;//make sure workspace has at least one block column

            /*Make LD and n multiple of 32*/
            //if(WORK_LD%32!=0) WORK_LD = ((WORK_LD + 31)/32)*32;
            //if(WORK_n%32!=0) WORK_n = ((WORK_n + 31)/32)*32;
            /*Allocate workspace*/
            if (MAGMA_SUCCESS != magma_dmalloc_pinned(&WORK, WORK_LD*WORK_n)) { 
            //if (MAGMA_SUCCESS != magma_dmalloc_cpu(&WORK, WORK_LD*WORK_n)) {
                info = MAGMA_ERR_HOST_ALLOC;
                printf("magma_dmalloc_pinned returned error %d: %s.\n     ", (int) info);
            }
            /*First touch the workspace with each thread. This may be needed to avoid using numactl --interleave*/
            //magma_amc_dmemset(WORK, 0.0, WORK_LD*WORK_n, 256, P); //nb
            //#pragma omp parallel for  private(info) schedule(static,nb)
            //for(info=0;info<WORK_LD*WORK_n;info++) WORK[info] = 0.0; //alternative first touch by the thread

            magma_amc_init(P, d_cpu, Pr, nb);

            gpu_time3 = magma_wtime();
            magma_dgetrf_gpu_work_amc_v2(M, N, d_A, ldda, ipiv, &info, WORK, WORK_LD, WORK_n);
            gpu_time3 = magma_wtime() - gpu_time3;
            gpu_perf3 = gflops / gpu_time3;
            magma_amc_finalize();

            if (info != 0)
                printf("magma_amc_dgetrf_rec_work_gpu returned error %d: %s.\n     ",
                       (int) info, magma_strerror( info ));

            /*Free workspace*/
            magma_free_pinned(WORK);
            //free(WORK);

            /*Restore BLAS number of threads*/
            if(lapack_numthreads !=1){ 
                /*Temporary set the number of threads to 1.*/
                printf("MAGMA num threads set from %d to %d\n", 1, lapack_numthreads);
                magma_setlapack_numthreads(lapack_numthreads);
            }

            /* =====================================================================
               Print results
               =================================================================== */
            printf("%5d %5d", (int) M, (int) N);
            if(cpu_perf!=0.0){
                printf("   %7.2f (%7.2f)", cpu_perf, cpu_time);
            }
            else{
                printf("   ---   (  ---  )");
            }
            if(gpu_perf1!=0.0){
                printf("   %7.2f (%7.2f)", gpu_perf1, gpu_time1);
            }
            else{
                printf("   ---   (  ---  )");
            }
            if(gpu_perf2!=0.0){
                printf("   %7.2f (%7.2f)", gpu_perf2, gpu_time2);
            }
            else{
                printf("   ---   (  ---  )");
            }
            if(gpu_perf3!=0.0){
                printf("        %7.2f (%7.2f)", gpu_perf3, gpu_time3);
            }
            else{
                printf("        ---   (  ---  )");
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            
            if ( opts.check ) {
                magma_dgetmatrix( M, N, d_A, ldda, h_A, lda );
                error = get_LU_error( M, N, h_R, lda, h_A, ipiv );
                printf("   %8.2e\n", error );
            }
            else {
                printf("     ---  \n");
            }
            printf("\n");
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
