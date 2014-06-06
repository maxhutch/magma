/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:56 2013
       
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

//#include "trace.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_chemm_mgpu
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex calpha    = MAGMA_C_MAKE( 3.456, 5.678 );
    magmaFloatComplex cbeta     = MAGMA_C_MAKE( 1.234, 2.456 );
    
    real_Double_t    gflops, gpu_perf=0., cpu_perf=0., gpu_time=0., cpu_time=0.;
    real_Double_t    gpu_perf2=0., gpu_time2=0.;
    float           error=0., errorbis=0., work[1];
    magmaFloatComplex *hA, *hX, *hB, *hR;
    magmaFloatComplex *dA[MagmaMaxGPUs], *dX[MagmaMaxGPUs], *dB[MagmaMaxGPUs], *dwork[MagmaMaxGPUs], *hwork[MagmaMaxGPUs+1];
    magmaFloatComplex *dA2;
    
    /* Matrix size */
    magma_int_t m, size, lda, ldda;
    const int MAXTESTS = 10;
    // sizes are 1024*N - 32
    magma_int_t msize[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };
    magma_int_t n       = 64;
    magma_int_t nb      = 64;
    int nstream = 3;
    int count   = 3;
    int ngpu    = magma_num_gpus();
    int ver =225;
    magma_int_t ione     = 1;
    magma_int_t iseed[4] = {0,0,0,1};
        
    printf( "Usage: %s -M m -N n -nb nb -nstream nstream -ngpu ngpu -count count -c\n"
            "    -M can be repeated %d times.\n"
            "    -ngpu or setting $MAGMA_NUM_GPUS sets number of GPUs to use.\n"
            "    -c or setting $MAGMA_TESTINGS_CHECK runs LAPACK and checks result.\n\n",
            argv[0], MAXTESTS );

    int checkres = (getenv("MAGMA_TESTINGS_CHECK") != NULL);

    int ntest = 0;
    int mmax = 0;
    for( int i = 1; i < argc; i++ ) {
        if ( strcmp("-M", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( ntest < MAXTESTS, "error: -M repeated more than maximum %d tests\n", MAXTESTS );
            msize[ntest] = atoi( argv[++i] );
            magma_assert( msize[ntest] > 0, "error: -M %s is invalid; must be > 0.\n", argv[i] );
            mmax = max( mmax, msize[ntest] );
            ntest++;
        }
        else if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            n = atoi( argv[++i] );
            magma_assert( n > 0, "error: -N %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-NB", argv[i]) == 0 && i+1 < argc ) {
            nb = atoi( argv[++i] );
            magma_assert( nb > 0, "error: -nb %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-v", argv[i]) == 0 && i+1 < argc ) {
            ver = atoi( argv[++i] );
            magma_assert( nb > 0, "error: -nb %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-count", argv[i]) == 0 && i+1 < argc ) {
            count = atoi( argv[++i] );
            magma_assert( count > 0, "error: -count %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-nstream", argv[i]) == 0 && i+1 < argc ) {
            nstream = atoi( argv[++i] );
            magma_assert( nstream > 0 && nstream <= 20,
                    "error: -nstream %s is invalid; must be > 0 and <= 20.\n", argv[i] );
        }
        else if ( strcmp("-ngpu", argv[i]) == 0 && i+1 < argc ) {
            ngpu = atoi( argv[++i] );
            magma_assert( ngpu > 0, "error: -ngpu %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-c", argv[i]) == 0 ) {
            checkres = true;
        }
        else {
            printf( "invalid argument: %s\n", argv[i] );
            //exit(1);
        }
    }
    if ( ntest == 0 ) {
        ntest = MAXTESTS;
        mmax = msize[ntest-1];
    }
    m = mmax;
    assert( m > 0 && n > 0 );
    
    // allocate memory for largest problem
    lda  = m;
    ldda = ((m + 31)/32)*32;

    
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2];
    magma_int_t nbcmplx=0;
    magma_buildconnection_mgpu(gnode, &nbcmplx,  ngpu);
    printf(" Initializin communication pattern.... GPU-ncmplx %d\n\n" , (int) nbcmplx);

    for (int i=0;i<nbcmplx;++i)
    {
        int myngpu =gnode[i][MagmaMaxGPUs];
        printf("cmplx %d has %d gpu ", i, myngpu);
        for(int j=0;j<myngpu;++j)
            printf("  %d", (int) gnode[i][j]);
        printf("\n");
    }

    TESTING_MALLOC_CPU( hA, magmaFloatComplex, lda*m );
    TESTING_MALLOC_CPU( hX, magmaFloatComplex, lda*n );
    TESTING_MALLOC_CPU( hB, magmaFloatComplex, lda*n );
    
    TESTING_MALLOC_PIN( hR, magmaFloatComplex, lda*n );

    magma_int_t  nbevents =2;
    cudaStream_t streams[MagmaMaxGPUs][20];
    cudaEvent_t  redevents[MagmaMaxGPUs][20];
    cudaEvent_t  redevents2[MagmaMaxGPUs][MagmaMaxGPUs*MagmaMaxGPUs+10];
    for( int d = 0; d < ngpu; ++d ) {
        magma_int_t mlocal = ((m / nb) / ngpu + 1) * nb;
        cudaSetDevice( d );
        TESTING_MALLOC_DEV( dA[d],    magmaFloatComplex, ldda*mlocal );
        TESTING_MALLOC_DEV( dX[d],    magmaFloatComplex, ldda*n      );
        TESTING_MALLOC_DEV( dB[d],    magmaFloatComplex, ldda*n      );
        TESTING_MALLOC_DEV( dwork[d], magmaFloatComplex, ldda*n*3    );
        
        TESTING_MALLOC_PIN( hwork[d], magmaFloatComplex, lda*n       );
        for( magma_int_t i = 0; i < nstream; ++i ) {
            cudaStreamCreate( &streams[d][i] );
        }
        for( magma_int_t i = 0; i < nbevents; ++i ) {
            cudaEventCreateWithFlags(&redevents[d][i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&redevents2[d][i], cudaEventDisableTiming);
        }
    }
    TESTING_MALLOC_PIN( hwork[ngpu], magmaFloatComplex, lda*n );



    if ( checkres ) {
        cudaSetDevice( 0 );
        TESTING_MALLOC_DEV( dA2, magmaFloatComplex, ldda*m );
    }
    
    printf( "nb %d, ngpu %d, nstream %d version %d \n", (int) nb, ngpu, nstream, ver );
    printf("    m     n    nb offset  CPU GFlop/s (sec)   GPU GFlop/s (sec)   CUBLAS hemm (sec)   ||R|| / ||A||*||X||\n");
    printf("=========================================================================================================\n");

//    for( int nb = 64; nb < 256; nb+=64 ) {
//            if(nb==192) nb=256;
//            printf("\n\n\n\n\n");

    magma_int_t nbtime=0;
    for( int i = 0; i < ntest; ++i ) {
    for( int offst = 0; offst < 1; offst += min(n,nb) ) {
    for( int j = 0; j < count; ++j ) {
        m = msize[i];
        assert( m > 0 && n > 0 );
        magma_int_t msiz = m-offst;

        lda  = m;
        ldda = ((m + 31)/32)*32;
        gflops = FLOPS_CHEMM( MagmaLeft, (float)msiz, (float)n ) / 1e9;

        size = lda*m;
        lapackf77_clarnv( &ione, iseed, &size, hA );
        magma_cmake_hermitian( m, hA, lda );
        
        size = lda*n;
        lapackf77_clarnv( &ione, iseed, &size, hX );
        lapackf77_clarnv( &ione, iseed, &size, hB );
        lapackf77_clacpy( "Full", &m, &n, hB, &lda, hR, &lda );
        
        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_csetmatrix_1D_col_bcyclic( m, m, hA, lda, dA, ldda, ngpu, nb );
        for( int d = 0; d < ngpu; ++d ) {
            cudaSetDevice( d );
            //magmablasSetKernelStream( streams[ d ][  0 ] );
            magma_csetmatrix( m, n, hX, lda, dX[d], ldda );
            //if(d==0) magma_csetmatrix( m, n, hB, lda, dB[d], ldda );// this is wrong coz when offset !=0 the gpu who do the beta*C may be not 0 so this should be related to stdev(starting device who own i=0 first col)
            magma_csetmatrix( m, n, hB, lda, dB[d], ldda );
        }
    



        //memset(hR, 0, lda*n*sizeof(magmaFloatComplex));

        //trace_init( 1, ngpu, nstream, (cudaStream_t*) streams );

        //magma_int_t offst =0;//nb;

        //cudaDeviceSynchronize();
        gpu_time = magma_wtime();
        // 1GPU version light
        /*
        magmablas_chemm_1gpu(
            MagmaLeft, MagmaLower, msiz, n,
            calpha,    dA, ldda, offst,
                       dX, ldda,
            cbeta,     dB, ldda, hR, lda,
            ngpu, nb, streams, nstream );
        */
        // multi gpu version
        
        if (ver==21) {
            // TODO: not available?
            //magmablas_chemm_mgpu(
            //    MagmaLeft, MagmaLower, msiz, n,
            //    calpha,    dA, ldda, offst,
            //               dX, ldda,
            //    cbeta,     dB, ldda, dwork, ldda, hR, lda, hwork, lda,
            //    ngpu, nb, streams, nstream, redevents, nbevents );
        }
        else {
            magmablas_chemm_mgpu_com(
                MagmaLeft, MagmaLower, msiz, n,
                calpha,    dA, ldda, offst,
                           dX, ldda,
                cbeta,     dB, ldda, dwork, ldda, hR, lda, hwork, lda,
                ngpu, nb, streams, nstream, redevents2, nbevents, gnode, nbcmplx);
        }
       
        cudaDeviceSynchronize();
        gpu_time = magma_wtime() - gpu_time;
        gpu_perf = gflops / gpu_time;
            
        #ifdef TRACING
        char buf[80];
        snprintf( buf, sizeof(buf), "chemm-m%d-n%d-nb%d-stream%d-ngpu%d-run%d.svg",
                  (int) m, (int) n, (int) nb, (int) nstream, (int) ngpu, (int) j );
        trace_finalize( buf, "trace.css" );
        #endif
        
        /* ====================================================================
           Performs operation using CUBLAS
           =================================================================== */
        if (( checkres )&&(nbtime==0)) {
            nbtime =1;
            magma_setdevice( 0 );
            magmablasSetKernelStream(  0  );
            magma_csetmatrix( m, m, hA, lda, dA2, ldda );
            magma_csetmatrix( m, n, hX, lda, dX[0], ldda );
            magma_csetmatrix( m, n, hB, lda, dwork[0], ldda );
            
            cudaDeviceSynchronize();
            gpu_time2 = magma_wtime();
            magma_chemm(
                MagmaLeft, MagmaLower, msiz, n,
                calpha,    dA2+offst*ldda+offst,   ldda,
                           dX[0], ldda,
                cbeta,     dwork[0], ldda );
            cudaDeviceSynchronize();
            gpu_time2 = magma_wtime() - gpu_time2;
            gpu_perf2 = gflops / gpu_time2;
        }
        
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */

        if ( checkres ) {
            // store ||A||*||X||
            errorbis  = lapackf77_clange("fro", &msiz, &msiz, hA+offst*lda+offst, &lda, work );
            errorbis *= lapackf77_clange("fro", &msiz, &n, hX, &lda, work );
            
            //printf( "A =" ); magma_cprint( m, m, hA, lda );
            //printf( "X =" ); magma_cprint( m, n, hX, lda );
            //printf( "B =" ); magma_cprint( m, n, hB, lda );
            
            cpu_time = magma_wtime();
            blasf77_chemm( "Left", "Lower", &msiz, &n,
                            &calpha,    hA+offst*lda+offst, &lda,
                                        hX, &lda,
                            &cbeta,     hB, &lda );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            /*
              trace_file = fopen("AJETE/C", "w");
              for (int j = 0; j < n ; j++)
                    for (int i = 0; i < siz ; i++)
                                   fprintf(trace_file, "%10d%10d%40.30e\n", i+1, j+1, hB[j*lda+i]);
              fclose(trace_file);
            */
            magma_int_t firstprint=0;
            for(magma_int_t dev=0; dev<ngpu; ++dev){
            
                magma_setdevice( dev );
                magma_cgetmatrix( m, n,  dB[dev], ldda, hR, lda );

                // compute relative error ||R||/||A||*||X||, where R := B_magma - B_lapack = R - B
                size = lda*n;
                blasf77_caxpy( &size, &c_neg_one, hB, &ione, hR, &ione );
                error = lapackf77_clange("fro", &msiz, &n, hR, &lda, work) / errorbis;
                
                //printf( "R ="  ); magma_cprint( m, n, hR, lda );
                if(firstprint==0)
                   printf( "%5d %5d %5d %5d   %7.1f (%7.4f)   %7.1f (%7.4f)   %7.1f (%7.4f)   %8.2e\n",
                        (int) m, (int) n, (int) nb, (int) offst,
                        cpu_perf, cpu_time,
                        gpu_perf, gpu_time,
                        gpu_perf2, gpu_time2, error );
                else
                   printf( "%89s  %8.2e\n", " ", error );
                firstprint =1;
             }
        } else {
            printf( "%5d %5d %5d %5d     ---   (  ---  )   %7.1f (%7.4f)     ---   (  ---  )   ---\n",
                    (int) m, (int) n, (int) nb, (int) offst,
                    /*cpu_perf, cpu_time,*/
                    gpu_perf, gpu_time
                    /*, gpu_perf2, gpu_time2, error*/ );
        }

    }}}//}
    
    /* Memory clean up */
    TESTING_FREE_CPU( hA );
    TESTING_FREE_CPU( hX );
    TESTING_FREE_CPU( hB );
    
    TESTING_FREE_PIN( hR );

    for( int d = 0; d < ngpu; ++d ) {
        cudaSetDevice( d );
        TESTING_FREE_DEV( dA[d]    );
        TESTING_FREE_DEV( dX[d]    );
        TESTING_FREE_DEV( dB[d]    );
        TESTING_FREE_DEV( dwork[d] );
        
        TESTING_FREE_PIN( hwork[d] );
        for( magma_int_t i = 0; i < nstream; ++i ) {
            cudaStreamDestroy( streams[d][i] );
        }
        for( magma_int_t i = 0; i < nbevents; ++i ) {
            cudaEventDestroy( redevents[d][i]  );
            cudaEventDestroy( redevents2[d][i] );
        }
    }
    TESTING_FREE_PIN( hwork[ngpu] );

    if ( checkres ) {
        cudaSetDevice( 0 );
        TESTING_FREE_DEV( dA2 );
    }
    
    /* Shutdown */
    TESTING_FINALIZE();
    return 0;
}
