/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated d Fri Sep 13 16:21:19 2013
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#include "magma_amc.h"

/* number of columns on device dev in a cyclic distribution of the data*/
static int numcols2p(int dev, int n, int b, int P)
{
/*
dev:    device id
n: Number of blocks
b: block size
P: Number of GPUS
*/
    int N, rest,ncols;
    N = (int) ((n - (n % b))/b);

    rest = N % P;
    ncols = (N-rest)/P;

    if(dev<rest) 
        ncols+=1;

    ncols = ncols * b;
    //if(n%b!=0 && dev==P-1) 
    if(n%b!=0 && dev == (N%P)) 
        ncols = ncols + (n%b);

    return ncols;
}
/*
static int numcols2p(int dev, int n, int b, int P)
{
    int N, rest,ncols;
    N = (int) ((n - (n % b))/b);

    rest = N % P;
    ncols = (N-rest)/P;

    if(dev<rest) ncols+=1;

    ncols = ncols * b;
    if(n%b!=0 && dev==P-1) 
        ncols = ncols -b + (n%b);

    return ncols;
}
*/

// Initialize matrix to random.
// Having this in separate function ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix( int m, int n, double *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
double get_residual(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    if ( m != n ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }
    
    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const magma_int_t ione = 1;
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    double *x, *b;
    
    // initialize RHS
    TESTING_MALLOC_CPU( x, double, n );
    TESTING_MALLOC_CPU( b, double, n );
    lapackf77_dlarnv( &ione, ISEED, &n, b );
    blasf77_dcopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    lapackf77_dgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if (info != 0)
        printf("lapackf77_dgetrs returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    
    // reset to original A
    init_matrix( m, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_dgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    double norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_dlange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_dlange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_dlange( "F", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_dprint( 1, n, b, 1 );
    
    TESTING_FREE_CPU( x );
    TESTING_FREE_CPU( b );
    
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%d\n", norm_r, norm_A, norm_x, n );
    return norm_r / (n * norm_A * norm_x);
}


// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
double get_LU_error(magma_int_t M, magma_int_t N,
                    double *LU, magma_int_t lda,
                    magma_int_t *ipiv)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    double alpha = MAGMA_D_ONE;
    double beta  = MAGMA_D_ZERO;
    double *A, *L, *U;
    double work[1], matnorm, residual;
    
    TESTING_MALLOC_CPU( A, double, lda*N    );
    TESTING_MALLOC_CPU( L, double, M*min_mn );
    TESTING_MALLOC_CPU( U, double, min_mn*N );
    memset( L, 0, M*min_mn*sizeof(double) );
    memset( U, 0, min_mn*N*sizeof(double) );

    // set to original A
    init_matrix( M, N, A, lda );
    lapackf77_dlaswp( &N, A, &lda, &ione, &min_mn, ipiv, &ione);
    
    // copy LU to L and U, and set diagonal to 1
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

    TESTING_FREE_CPU(A);
    TESTING_FREE_CPU(L);
    TESTING_FREE_CPU(U);

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgetrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    real_Double_t   gpu_perf1, gpu_time1, gpu_perf2, gpu_time2, gpu_perf3, gpu_time3, alloc_time, free_time;
    double           error;
    double *h_A;
    double *d_lA[ MagmaMaxGPUs ];
    magma_int_t *ipiv;
    magma_int_t M, N, n2, lda, ldda, n_local, ngpu, NB;
    magma_int_t info, min_mn, nb, ldn_local;
    magma_int_t status = 0;

    magma_int_t P=-1;    /*Number of threads in the CPU part*/
    double d_cpu=-1;    /*pourcentgae of the matrix to allocate in the cpu part*/
    magma_int_t Pr=-1;  /*Number of threads for the panel*/
    magma_int_t async_nb; /*Block size*/
    
    double *WORK;
    magma_int_t WORK_LD, WORK_n;

    double **dlpanelT;
    magma_int_t dlpanelT_m, dlpanelT_n;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    P =  opts.nthread;
    async_nb = opts.nb;
    Pr = opts.panel_nthread;
    
    d_cpu = 0.0;
    #if defined(CPU_PEAK) && defined(GPU_PEAK)
    d_cpu = magma_amc_recommanded_dcpu(opts.nthread, CPU_PEAK, opts.ngpu, GPU_PEAK);
    #endif
    if(opts.fraction_dcpu!=0){ /*Overwrite the one computed with the model*/
    d_cpu = opts.fraction_dcpu;
    }
    magma_assert(d_cpu > 0 && d_cpu<=1.0,
    "error: The cpu fraction is invalid. Ensure you use --fraction_dcpu with fraction_dcpu in [0.0, 1.0] or compile with both -DCPU_PEAK=<cpu peak performance> and -DGPU_PEAK=<gpu peak performance> set.\n");
    
    
    printf("Asynchronous recursif LU... nb:%d, nbcores:%d, dcpu:%f, panel_nbcores:%d, ngpu: %d\n", async_nb, P, d_cpu, Pr, opts.ngpu);
    printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   GPU_Async_v2 GFlop/s (sec)  GPU_Async_work_v2 GFlop/s (sec)");
    if ( opts.check == 2 ) {
        printf("   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("   |PA-LU|/(N*|A|)\n");
    }
    printf("=========================================================================\n");

    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            //nb     = magma_get_dgetrf_nb( M );
            gflops = FLOPS_DGETRF( M, N ) / 1e9;
            
            
            
            // Allocate host memory for the matrix
            TESTING_MALLOC_CPU(    ipiv, magma_int_t,        min_mn );
            TESTING_MALLOC_CPU(    h_A,  double, n2     );
            
            /*set default number of threads for lapack*/
            magma_setlapack_numthreads(P);
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                init_matrix( M, N, h_A, lda );
                
                cpu_time = magma_wtime();
                lapackf77_dgetrf( &M, &N, h_A, &lda, ipiv, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dgetrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            
            nb = magma_get_dgetrf_nb( M );            

            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, int((N+nb-1)/nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) ngpu );
            }

            // Allocate device memory
            for( int dev=0; dev < ngpu; dev++){
                n_local = ((N/nb)/ngpu)*nb;
                if (dev < (N/nb) % ngpu)
                    n_local += nb;
                else if (dev == (N/nb) % ngpu)
                    n_local += N % nb;
                ldn_local = ((n_local+31)/32)*32;  // TODO why?
                magma_setdevice( dev );
                TESTING_MALLOC_DEV( d_lA[dev], double, ldda*ldn_local );
            }

            init_matrix( M, N, h_A, lda );
            magma_dsetmatrix_1D_col_bcyclic( M, N, h_A, lda, d_lA, ldda, ngpu, nb );
    
            gpu_time1 = magma_wtime();
            magma_dgetrf_mgpu( ngpu, M, N, d_lA, ldda, ipiv, &info );
            gpu_time1 = magma_wtime() - gpu_time1;
            gpu_perf1 = gflops / gpu_time1;
            if (info != 0)
                printf("magma_dgetrf_mgpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
                       
            magma_dgetmatrix_1D_col_bcyclic( M, N, d_lA, ldda, h_A, lda, ngpu, nb );

            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                TESTING_FREE_DEV( d_lA[dev] );
            }
            
            /* ====================================================================
               Performs operation using MAGMA_Async: This interface allocate workspace internally
               =================================================================== */

            /*For the benchmark we have 2 approaches*/
            /*1. use directly magma_amc */
            /*2. use magma_amc_work and add pinned memory time*/
            /*We choose approach 2*/
            /*
            nb = async_nb;

            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, int((N+nb-1)/nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) ngpu );
            }

            // Allocate device memory
            n_local = numcols2p(0, N, nb, ngpu);
            ldn_local = n_local;

            //ldn_local = ((n_local+31)/32)*32;
            for( int dev=0; dev < ngpu; dev++){
  
                magma_setdevice( dev );
                TESTING_MALLOC_DEV( d_lA[dev], double, ldda*ldn_local );
            }

            init_matrix( M, N, h_A, lda );
            magma_dsetmatrix_1D_col_bcyclic( M, N, h_A, lda, d_lA, ldda, ngpu, nb );
            
            // Switch to the sequential version of BLAS
            magma_setlapack_numthreads(1);

            magma_amc_init(P, d_cpu, Pr, nb);
            gpu_time2 = magma_wtime();
            magma_dgetrf_async_mgpu( ngpu, M, N, d_lA, ldda, ipiv, &info );
            gpu_time2 = magma_wtime() - gpu_time2;
            gpu_perf2 = gflops / gpu_time2;
            magma_amc_finalize();
            if (info != 0)
                printf("magma_dgetrf_mgpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
                       
            magma_dgetmatrix_1D_col_bcyclic( M, N, d_lA, ldda, h_A, lda, ngpu, nb );

            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                TESTING_FREE_DEV( d_lA[dev] );
            }
            */

            /* ====================================================================
               Performs operation using MAGMA_Async_Work
               =================================================================== */
            
            nb = async_nb;

            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, int((N+nb-1)/nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) ngpu );
            }

            // Allocate device memory
            n_local = numcols2p(0, N, nb, ngpu);
            ldn_local = n_local;

            //ldn_local = ((n_local+31)/32)*32;
            for( int dev=0; dev < ngpu; dev++){
                magma_setdevice( dev );
                TESTING_MALLOC_DEV( d_lA[dev], double, ldda*ldn_local );
            }

            init_matrix( M, N, h_A, lda );
            magma_dsetmatrix_1D_col_bcyclic( M, N, h_A, lda, d_lA, ldda, ngpu, nb );
            
            // Switch to the sequential version of BLAS
            magma_setlapack_numthreads(1);

            //Compute workspace dimension
            WORK_LD = M;

          
            NB  = (int) ceil( (double) N / nb);

            WORK_n = (int) ceil(N*d_cpu)+nb; /*TODO:remove +nb replace with A_N*/
            //WORK_n = NSplit(NB, d_cpu)*nb;

            if(WORK_n<nb) WORK_n = nb;//make sure workspace has at least one block column

            //Make LD and n multiple of 32
            //if(WORK_LD%32!=0) WORK_LD = ((WORK_LD + 31)/32)*32;
            //if(WORK_n%32!=0) WORK_n = ((WORK_n + 31)/32)*32;
            //Allocate workspace
            alloc_time = magma_wtime();
            if (MAGMA_SUCCESS != magma_dmalloc_pinned(&WORK, WORK_LD*WORK_n)) { 
            //if (MAGMA_SUCCESS != magma_dmalloc_cpu(&WORK, WORK_LD*WORK_n)) {
                info = MAGMA_ERR_HOST_ALLOC;
                printf("magma_dmalloc_pinned returned error %d: %s.\n     ", (int) info);
            }

            /* Workspace for the panels on the GPU*/
            dlpanelT_m = WORK_n; /*assume that the cpu and gpu use the same buffer size*/
            dlpanelT_n = M;
             dlpanelT = (double **)    malloc(ngpu*sizeof(double*));
              for(int dev=0;dev<ngpu;dev++){
                 magma_setdevice(dev);

                 if (MAGMA_SUCCESS != magma_dmalloc(&dlpanelT[dev], dlpanelT_m*dlpanelT_n)) { 

                
                        info = MAGMA_ERR_DEVICE_ALLOC; 
                        printf("magma_dmalloc returned error %d: %s.\n     ", (int) info);
                }
              }

            alloc_time = magma_wtime() - alloc_time;

            //First touch the workspace with each thread. This may be needed to avoid using numactl --interleave
            //magma_amc_dmemset(WORK, 0.0, WORK_LD*WORK_n, 256, P); //nb
            //#pragma omp parallel for  private(info) schedule(static,nb)
            //for(info=0;info<WORK_LD*WORK_n;info++) WORK[info] = 0.0; //alternative first touch by the thread

            magma_amc_init(P, d_cpu, Pr, nb);
            gpu_time3 = magma_wtime();
            magma_dgetrf_mgpu_work_amc_v3(ngpu, M, N, d_lA, ldda, ipiv, &info, WORK, WORK_LD, WORK_n);
            gpu_time3 = magma_wtime() - gpu_time3;
            gpu_perf3 = gflops / gpu_time3;
            magma_amc_finalize();
            if (info != 0)
                printf("magma_dgetrf_mgpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
                       
            magma_dgetmatrix_1D_col_bcyclic( M, N, d_lA, ldda, h_A, lda, ngpu, nb );

            

            //Free workspace
            free_time = magma_wtime();
            magma_free_pinned(WORK);
            
            for(int dev=0;dev<ngpu;dev++){
                magma_setdevice(dev);
                magma_free(dlpanelT[dev]);
            }

            free(dlpanelT);
            free_time = magma_wtime() - free_time;

            /*DEDUCE t2, JUST FOR THE BENCHMARK*/
             gpu_time2 =  gpu_time3 + alloc_time +  free_time;
             gpu_perf2 = gflops / gpu_time2;

            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                TESTING_FREE_DEV( d_lA[dev] );
            }
            /* =====================================================================
               Check the factorization
               =================================================================== */
            /*
            if ( opts.lapack ) {
                printf("%5d %5d  %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d %5d    ---   (  ---  )   %7.2f (%7.2f)",
                       (int) M, (int) N, gpu_perf, gpu_time );
            }
            */
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
                printf("   %7.2f (%7.2f)", gpu_perf3, gpu_time3);
            }
            else{
                printf("   ---   (  ---  )");
            }

            if ( opts.check == 2 ) {
                error = get_residual( M, N, h_A, lda, ipiv );
                printf("   %8.2e%s\n", error, (error < tol ? "" : "  failed"));
                status |= ! (error < tol);
            }
            else if ( opts.check ) {
                error = get_LU_error( M, N, h_A, lda, ipiv );
                printf("   %8.2e%s\n", error, (error < tol ? "" : "  failed"));
                status |= ! (error < tol);
            }
            else {
                printf( "     ---\n" );
            }
            
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A );
  
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
