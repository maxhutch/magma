/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @author Azzam Haidar

       @generated c Tue Dec 17 13:18:57 2013

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
#include "magma_cbulge.h"
#include "testings.h"
#include "magma_threadsetting.h"

#if defined(USEMKL)
#include <mkl_service.h>
#endif
#if defined(USEACML)
#include <omp.h>
#endif

// Flops formula
#define PRECISION_c
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_HETRD(n) + 2. * FADDS_HETRD(n))
#else
#define FLOPS(n) (      FMULS_HETRD(n) +      FADDS_HETRD(n))
#endif


#if defined(PRECISION_z) || defined(PRECISION_d)
extern "C" void cmp_vals(int n, float *wr1, float *wr2, float *nrmI, float *nrm1, float *nrm2);
extern "C" void ccheck_eig_(char *JOBZ, int  *MATYPE, int  *N, int  *NB,
                       magmaFloatComplex* A, int  *LDA, float *AD, float *AE, float *D1, float *EIG,
                    magmaFloatComplex *Z, int  *LDZ, magmaFloatComplex *WORK, float *RWORK, float *RESU);
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chetrd_he2hb
*/
int main( int argc, char** argv)
{
    TESTING_INIT_MGPU();

    magma_timestr_t       start, end;  //, tband;
    float           eps, flops, gpu_perf, gpu_time;
    magmaFloatComplex *h_A, *h_R, *h_work, *dT1;
    magmaFloatComplex *tau;
    float *D, *E;

    /* Matrix size */
    magma_int_t N = 0, n2, lda, lwork, ldt, lwork0;
    magma_int_t size[10] = {1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112};

    //magma_int_t j, k;
    magma_int_t i, info, checkres, once = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    const char *uplo = MagmaLowerStr;

    magma_int_t WANTZ=0;
    magma_int_t THREADS=1;
    magma_int_t NE = 0;
    magma_int_t NB = 0;
    magma_int_t ngpu = 1;

    checkres  = 0; //getenv("MAGMA_TESTINGS_CHECK") != NULL;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
                once = 1;
            }
            else if (strcmp("-NB", argv[i])==0) {
                NB = atoi(argv[++i]);
            }
            else if (strcmp("-threads", argv[i])==0) {
                THREADS = atoi(argv[++i]);
            }
            else if (strcmp("-ngpu", argv[i])==0) {
                ngpu = atoi(argv[++i]);
            }
            else if (strcmp("-wantz", argv[i])==0) {
                WANTZ = atoi(argv[++i]);
            }
            else if (strcmp("-NE", argv[i])==0) {
                NE = atoi(argv[++i]);
            }
            else if ( strcmp("-c", argv[i]) == 0 ) {
                checkres = 1;
            }
            else if (strcmp("-U", argv[i])==0)
                uplo = MagmaUpperStr;
            else if (strcmp("-L", argv[i])==0)
                uplo = MagmaLowerStr;
        }
        if ( N > 0 )
            printf("  testing_chetrd_he2hb -L|U -N %d -NB %d   -wantz %d   -threads %d    check %d \n\n",
                   (int) N, (int) NB, (int) WANTZ, (int) THREADS, (int) checkres);
        else
        {
            printf("\nUsage: \n");
            printf("  testing_chetrd_he2hb -L|U -N %d -NB  -wantz -threads \n\n", 1024);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_chetrd_he2hb -L|U -N %d\n\n", 1024);
        N = size[9];
    }
    
    eps = lapackf77_slamch( "E" );
    lda = N;
    ldt = N;
    n2  = lda * N;
    if(NB<1)
        NB  = 64; //64; //magma_get_chetrd_he2hb_nb(N);

    if(NE<1)
        NE  = N; //64; //magma_get_chetrd_he2hb_nb(N);

    /* We suppose the magma NB is bigger than lapack NB */
    lwork0 = N*NB;

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, lda*N  );
    TESTING_MALLOC_CPU( tau,    magmaFloatComplex, N-1    );
    
    TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, lda*N  );
    TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lwork0 );
    TESTING_MALLOC_PIN( D, float, N );
    TESTING_MALLOC_PIN( E, float, N );
    
    //TESTING_MALLOC_DEV( dT1, magmaFloatComplex, (2*min(N,N)+(N+31)/32*32)*NB );
    TESTING_MALLOC_DEV( dT1, magmaFloatComplex, (N*NB) );

    printf("  N    GPU GFlop/s   \n");
    printf("=====================\n");
    for(i=0; i<10; i++){
        if ( !once ) {
            N = size[i];
        }
        lda  = N;
        n2   = N*lda;
        flops = FLOPS_CHETRD( (float)N ) / 1e6;
        
        // if(WANTZ) flops = 2.0*flops;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        magma_cmake_hermitian( N, h_A, lda );
        
        lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );


        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_device_t cdev;
        magma_getdevice( &cdev );

        start = get_current_time();
        /*
        magma_chetrd_he2hb(uplo[0], N, NB, h_R, lda, tau, h_work, lwork0, dT1, THREADS, &info);
        tband = get_current_time();
        printf("  Finish BAND  N %d  NB %d  ngpu %d timing= %f\n", N, NB, ngpu, GetTimerValue(start, tband) / 1000.);
        magma_chetrd_bhe2trc_v5(THREADS, WANTZ, uplo[0], NE, N, NB, h_R, lda, D, E, dT1, ldt);
        */

        /*
        magma_chetrd_he2hb(uplo[0], N, NB, h_R, lda, tau, h_work, lwork, dT1, THREADS, &info);
        tband = get_current_time();
        printf("  Finish BAND  N %d  NB %d  ngpu %d timing= %f\n", N, NB, ngpu, GetTimerValue(start, tband) / 1000.);
        magma_chetrd_bhe2trc(THREADS, WANTZ, uplo[0], NE, N, NB, h_R, lda, D, E, dT1, ldt);
        */

        char *jobz = (char*)MagmaVecStr;
        char range = 'A';
        magma_int_t fraction_ev = 100;
        magma_int_t il, iu, m1;
        float vl=0., vu=0.;

        if (fraction_ev == 0){
            il = N / 10;
            iu = N / 5+il;
        }
        else {
            il = 1;
            iu = (int)(fraction_ev*N);
            if (iu < 1) iu = 1;
        }
        magmaFloatComplex *hh_work;
        magma_int_t *iwork;
        magma_int_t nb, /*lwork,*/ liwork;
        magma_int_t threads = magma_get_numthreads();
#if defined(PRECISION_z) || defined(PRECISION_c)
        float *rwork;
        magma_int_t lrwork;
        lwork  = magma_cbulge_get_lq2(N, threads) + 2*N + N*N;
        lrwork = 1 + 5*N +2*N*N;
        TESTING_MALLOC_PIN( rwork, float, lrwork );
#else
        lwork  = magma_cbulge_get_lq2(N, threads) + 1 + 6*N + 2*N*N;
#endif
        liwork = 3 + 5*N;
        nb = magma_get_chetrd_nb(N);
        TESTING_MALLOC_PIN( hh_work, magmaFloatComplex, lwork  );
        TESTING_MALLOC_CPU( iwork,   magma_int_t,        liwork );

        if(ngpu==1){
            printf("calling cheevdx_2stage 1 GPU\n");
            magma_cheevdx_2stage(jobz[0], range, uplo[0], N, 
                            h_R, lda, 
                            vl, vu, il, iu, 
                            &m1, D, 
                            hh_work, lwork, 
#if defined(PRECISION_z) || defined(PRECISION_c)
                            rwork, lrwork, 
#endif
                            iwork, liwork, 
                            &info);

        }else{
            printf("calling cheevdx_2stage_m %d GPU\n", (int) ngpu);
            magma_cheevdx_2stage_m(ngpu, jobz[0], range, uplo[0], N, 
                            h_R, lda, 
                            vl, vu, il, iu, 
                            &m1, D, 
                            hh_work, lwork, 
#if defined(PRECISION_z) || defined(PRECISION_c)
                            rwork, lrwork, 
#endif
                            iwork, liwork, 
                            &info);
        }


        magma_setdevice( cdev );        
        end = get_current_time();


        gpu_perf = flops / GetTimerValue(start, end);
        gpu_time = GetTimerValue(start, end) / 1000.;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        /*
        if ( checkres ) {
            FILE        *fp ;

            printf("Writing input matrix in matlab_i_mat.txt ...\n");
            fp = fopen ("matlab_i_mat.txt", "w") ;
            if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

            for(j=0; j<N; j++) {
                for(k=0; k<N; k++) {
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    fprintf(fp, "%5d %5d %11.8f %11.8f\n", k+1, j+1,
                            h_A[k+j*lda].x, h_A[k+j*lda].y);
                    #else
                    fprintf(fp, "%5d %5d %11.8f\n", k+1, j+1, h_A[k+j*lda]);
                    #endif
                }
            }
            fclose( fp ) ;

            printf("Writing output matrix in matlab_o_mat.txt ...\n");
            fp = fopen ("matlab_o_mat.txt", "w") ;
            if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

            for(j=0; j<N; j++) {
                for(k=0; k<N; k++) {
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    fprintf(fp, "%5d %5d %11.8f %11.8f\n", k+1, j+1,
                            h_R[k+j*lda].x, h_R[k+j*lda].y);
                    #else
                    fprintf(fp, "%5d %5d %11.8f\n", k+1, j+1, h_R[k+j*lda]);
                    #endif
                }
            }
            fclose( fp ) ;
        }
        */



        /* =====================================================================
           Print performance and error.
           =================================================================== */
#if defined(CHECKEIG)
#if defined(PRECISION_z)  || defined(PRECISION_d)
        if ( checkres ) {
            printf("  Total N %5d  flops %6.2f  timing %6.2f seconds\n", (int) N, gpu_perf, gpu_time );
            char JOBZ;
            if(WANTZ==0)
                    JOBZ='N';
            else
                    JOBZ = 'V';
            float nrmI=0.0, nrm1=0.0, nrm2=0.0;
            int    lwork2 = 256*N;
            magmaFloatComplex *work2     = (magmaFloatComplex *) malloc (lwork2*sizeof(magmaFloatComplex));
            float *rwork2     = (float *) malloc (N*sizeof(float));
            float *D2          = (float *) malloc (N*sizeof(float));
            magmaFloatComplex *AINIT    = (magmaFloatComplex *) malloc (N*lda*sizeof(magmaFloatComplex));
            memcpy(AINIT, h_A, N*lda*sizeof(magmaFloatComplex));
            /* compute the eigenvalues using lapack routine to be able to compare to it and used as ref */
            start = get_current_time();
            i= min(12, THREADS);

            #if defined(USEMKL)
            mkl_set_num_threads( i );
            #endif
            #if defined(USEACML)
            omp_set_num_threads(i);
            #endif

            #if defined(PRECISION_z) || defined (PRECISION_c)
            lapackf77_cheev( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, rwork2, &info );
            #else
            lapackf77_ssyev( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, &info );
            #endif
            ///* call eigensolver for our resulting tridiag [D E] and for Q */
            //dstedc_withZ('V', N, D, E, h_R, lda);
            ////ssterf_( &N, D, E, &info);
            ////
            end = get_current_time();
            printf("  Finish CHECK - EIGEN   timing= %f  threads %d\n", GetTimerValue(start, end) / 1000., i);

            /*
            for(i=0;i<10;i++)
                printf(" voici lpk D[%d] %8.2e\n", i, D2[i]);
            */

            //magmaFloatComplex mydz=0.0, mydo=1.0;
            //magmaFloatComplex *Z = (magmaFloatComplex *) malloc(N*lda*sizeof(magmaFloatComplex));
            // dgemm_("N", "N", &N, &N, &N, &mydo, h_R, &lda, h_A, &lda, &mydz, Z, &lda);


            /* compare result */
            cmp_vals(N, D2, D, &nrmI, &nrm1, &nrm2);


            magmaFloatComplex *WORKAJETER;
            float *RWORKAJETER, *RESU;
            WORKAJETER  = (magmaFloatComplex *) malloc( (2* N * N + N) * sizeof(magmaFloatComplex) );
            RWORKAJETER = (float *) malloc( N * sizeof(float) );
            RESU        = (float *) malloc(10*sizeof(float));
            int MATYPE;
            memset(RESU, 0, 10*sizeof(float));

 
            MATYPE=3;
            float NOTHING=0.0;
            start = get_current_time();
            // check results
            ccheck_eig_(&JOBZ, &MATYPE, &N, &NB, AINIT, &lda, &NOTHING, &NOTHING, D2, D, h_R, &lda, WORKAJETER, RWORKAJETER, RESU );
            end = get_current_time();
            printf("  Finish CHECK - results timing= %f\n", GetTimerValue(start, end) / 1000.);
            #if defined(USEMKL)
            mkl_set_num_threads( 1 );
            #endif
            #if defined(USEACML)
            omp_set_num_threads(1);
            #endif

            printf("\n");
            printf(" ================================================================================================================\n");
            printf("   ==> INFO voici  threads=%d    N=%d    NB=%d   WANTZ=%d\n", (int) THREADS, (int) N, (int) NB, (int) WANTZ);
            printf(" ================================================================================================================\n");
            printf("            DSBTRD                : %15s \n", "STATblgv9withQ    ");
            printf(" ================================================================================================================\n");
            if(WANTZ>0)
                printf(" | A - U S U' | / ( |A| n ulp )   : %15.3E   \n", RESU[0]);
            if(WANTZ>0)
                printf(" | I - U U' | / ( n ulp )         : %15.3E   \n", RESU[1]);
            printf(" | D1 - EVEIGS | / (|D| ulp)      : %15.3E   \n",  RESU[2]);
            printf(" max | D1 - EVEIGS |              : %15.3E   \n",  RESU[6]);
            printf(" ================================================================================================================\n\n\n");
            
            printf(" ****************************************************************************************************************\n");
            printf(" * Hello here are the norm  Infinite (max)=%8.2e  norm one (sum)=%8.2e   norm2(sqrt)=%8.2e *\n", nrmI, nrm1, nrm2);
            printf(" ****************************************************************************************************************\n\n");
        }
#endif
#endif

        printf("  Total N %5d  flops %6.2f        timing %6.2f seconds\n", (int) N, gpu_perf, gpu_time );
        printf("============================================================================\n\n\n");

        if ( once )
            break;
    }

    /* Memory clean up */
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_CPU( tau );
    
    TESTING_FREE_PIN( h_R    );
    TESTING_FREE_PIN( h_work );
    TESTING_FREE_PIN( D      );
    TESTING_FREE_PIN( E      );
    
    TESTING_FREE_DEV( dT1 );

    /* TODO - not all memory has been freed inside loop */
    
    /* Shutdown */
    TESTING_FINALIZE_MGPU();
    return EXIT_SUCCESS;
}
