/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Stan Tomov

       @precisions normal z -> s d c

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

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhetrd
*/

int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf=0., cpu_perf=0., gpu_time=0., cpu_time=0.;
    double           eps;
    cuDoubleComplex *h_A, *h_R, *h_Q, *h_work, *work;
    cuDoubleComplex *tau;
    double          *diag, *offdiag, *rwork;
    double           result[2] = {0., 0.};
    int num_gpus = 0;

    /* Matrix size */
    magma_int_t N = 0, n2, lda, lwork;
    const int MAXTESTS = 10;
    //magma_int_t size[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };
    magma_int_t size[MAXTESTS] = { 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000 };

    magma_int_t i, k = 1, info, nb, checkres;
    magma_int_t ione     = 1;
    magma_int_t itwo     = 2;
    magma_int_t ithree   = 3;
    magma_int_t ISEED[4] = {0,0,0,1};
    const char *uplo = MagmaLowerStr;

    checkres  = getenv("MAGMA_TESTINGS_CHECK") != NULL;
   

    printf( "\nUsage: %s -N <matrix size> -R <right hand sides> [-L|-U] -c\n", argv[0] );
    printf( "  -N can be repeated up to %d times\n", MAXTESTS );
    printf( "  -c or setting $MAGMA_TESTINGS_CHECK checks result.\n\n" );
    int ntest = 0;
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( ntest < MAXTESTS, "error: -N repeated more than maximum %d tests\n", MAXTESTS );
            size[ntest] = atoi( argv[++i] );
            magma_assert( size[ntest] > 0, "error: -N %s is invalid; must be > 0.\n", argv[i] );
            N = max( N, size[ntest] );
            ntest++;
        }
        else if ( strcmp("-L", argv[i]) == 0 ) 
            uplo = MagmaLowerStr;
        else if ( strcmp("-U", argv[i]) == 0 ) 
            uplo = MagmaUpperStr;
        else if (strcmp("-K",argv[i])==0)
                k = atoi(argv[++i]);
        else if ( strcmp("-c", argv[i]) == 0 ) 
            checkres = true;
        else if (strcmp("-NGPU",argv[i])==0)
            num_gpus = atoi(argv[++i]);
        else {
            printf( "invalid argument: %s\n", argv[i] );
            exit(1);
        }
    }
    printf( " num_gpus = %d\n",num_gpus );
    if ( ntest == 0 ) {
        ntest = 6; //MAXTESTS;
        N = size[ntest-1];
    }

    eps = lapackf77_dlamch( "E" );
    lda = N;
    n2  = lda * N;
    nb  = magma_get_zhetrd_nb(N);
   

    /* We suppose the magma nb is bigger than lapack nb */
    lwork = N*nb; 

    /* Allocate host memory for the matrix */
    TESTING_MALLOC(    h_A,    cuDoubleComplex, lda*N );
    TESTING_HOSTALLOC( h_R,    cuDoubleComplex, lda*N );
    TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork );
    TESTING_MALLOC(    tau,    cuDoubleComplex, N     );
    TESTING_MALLOC( diag,    double, N   );
    TESTING_MALLOC( offdiag, double, N-1 );

    /* To avoid uninitialized variable warning */
    h_Q   = NULL;
    work  = NULL;
    rwork = NULL; 

    if ( checkres ) {
        TESTING_MALLOC( h_Q,  cuDoubleComplex, lda*N );
        TESTING_MALLOC( work, cuDoubleComplex, 2*N*N );
#if defined(PRECISION_z) || defined(PRECISION_c) 
        TESTING_MALLOC( rwork, double, N );
#endif
    }

    printf("  N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   |A-QHQ'|/N|A|   |I-QQ'|/N\n");
    printf("===========================================================================\n");
    for( i = 0; i < ntest; ++i ) {
        N = size[i];
        lda  = N;
        n2   = N*lda;
        gflops = FLOPS_ZHETRD( (double)N ) / 1e9;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        magma_zmake_hermitian( N, h_A, lda );
        
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        gpu_time = magma_wtime();
        if( num_gpus == 0 ) {
            magma_zhetrd(uplo[0], N, h_R, lda, diag, offdiag, 
                         tau, h_work, lwork, &info);
        } else {
            magma_zhetrd_mgpu(num_gpus, k, uplo[0], N, h_R, lda, diag, offdiag,
                              tau, h_work, lwork, &info);
        }
        gpu_time = magma_wtime() - gpu_time;
        if ( info != 0 )
            printf("magma_zhetrd returned error %d\n", (int) info);

        gpu_perf = gflops / gpu_time;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        if ( checkres ) {
            lapackf77_zlacpy(uplo, &N, &N, h_R, &lda, h_Q, &lda);
            lapackf77_zungtr(uplo, &N, h_Q, &lda, tau, h_work, &lwork, &info);

#if defined(PRECISION_z) || defined(PRECISION_c) 
            lapackf77_zhet21(&itwo, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, rwork, &result[0]);

            lapackf77_zhet21(&ithree, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, rwork, &result[1]);

#else

            lapackf77_zhet21(&itwo, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, &result[0]);

            lapackf77_zhet21(&ithree, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, &result[1]);

#endif
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_zhetrd(uplo, &N, h_A, &lda, diag, offdiag, tau, 
                         h_work, &lwork, &info);
        cpu_time = magma_wtime() - cpu_time;
        if ( info != 0 )
            printf("lapackf77_zhetrd returned error %d\n", (int) info);

        cpu_perf = gflops / cpu_time;

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        if ( checkres ) {
            printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e        %8.2e\n",
                   (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                   result[0]*eps, result[1]*eps );
        } else {
            printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)     ---  \n",
                   (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
        }
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_FREE( tau );
    TESTING_FREE( diag );
    TESTING_FREE( offdiag );
    TESTING_HOSTFREE( h_R );
    TESTING_HOSTFREE( h_work );

    if ( checkres ) {
        TESTING_FREE( h_Q );
        TESTING_FREE( work );
#if defined(PRECISION_z) || defined(PRECISION_c) 
        TESTING_FREE( rwork );
#endif
    }

    /* Shutdown */
    TESTING_FINALIZE();
    return EXIT_SUCCESS;
}
