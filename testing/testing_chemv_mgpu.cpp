/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:56 2013
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cblas.h>

#include "flops.h"
#include "magma.h"
#include "magmablas.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_c

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_SYMV(n) + 2. * FADDS_SYMV(n))
#else
#define FLOPS(n) (      FMULS_SYMV(n) +      FADDS_SYMV(n))
#endif

#define MultiGPUs
#define validate


int main(int argc, char **argv)
{        
    TESTING_INIT();
    magma_setdevice(0);

    magma_timestr_t  start, end;
    float      flops, magma_perf, cuda_perf, error, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t n_local[4];

    FILE        *fp ; 
    magma_int_t N, m, i, j, lda, LDA, M;
    magma_int_t matsize;
    magma_int_t vecsize;
    magma_int_t istart = 64;
    magma_int_t incx = 1;
    char        uplo = MagmaLower;

    magmaFloatComplex alpha = MAGMA_C_MAKE(1., 0.); // MAGMA_C_MAKE(  1.5, -2.3 );
    magmaFloatComplex beta  = MAGMA_C_MAKE(0., 0.); // MAGMA_C_MAKE( -0.6,  0.8 );
    magmaFloatComplex *A, *X, *Y[4], *Ycublas, *Ymagma;
    magmaFloatComplex *dA, *dX[4], *dY[4], *d_lA[4], *dYcublas ;

    magma_queue_t stream[4][10];
    magmaFloatComplex *C_work;
    magmaFloatComplex *dC_work[4];

    int max_num_gpus;
    magma_int_t num_gpus = 1, nb;
    magma_int_t blocks, lwork;
    magma_int_t offset = 0;

    M = 0;
    N = 0;
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
            {
                N = atoi(argv[++i]);
                istart = N;
            }
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
            else if (strcmp("-NGPU", argv[i])==0)
              num_gpus = atoi(argv[++i]);
            else if (strcmp("-offset", argv[i])==0)
              offset = atoi(argv[++i]);
        }
        if ( M == 0 ) {
            M = N;
        }
        if ( N == 0 ) {
            N = M;
        }
        if (M>0 && N>0)
        {    printf("  testing_chemv_mgpu -M %d -N %d -NGPU %d\n\n", (int) M, (int) N, (int) num_gpus);
            printf("  in %c side \n", uplo);
        }
        else
            {
                printf("\nUsage: \n");
                printf("  testing_chemv_mgpu -M %d -N %d -NGPU %d\n\n", 
                       1024, 1024, 1);
                exit(1);
            }
    }
    else {
#if defined(PRECISION_z)
        M = N = 8000;
#else
        M = N = 12480;
#endif 
        num_gpus = 2;
        offset = 0;
        printf("\nUsage: \n");
        printf("  testing_chemv_mgpu -M %d -N %d -NGPU %d\n\n", (int) M, (int) N, (int) num_gpus);
    }
         

    //////////////////////////////////////////////////////////////////////////
    cudaGetDeviceCount(&max_num_gpus);
    if (num_gpus > max_num_gpus){
      printf("More GPUs requested than available. Have to change it.\n");
      num_gpus = max_num_gpus;
    }
    printf("Number of GPUs to be used = %d\n", (int) num_gpus);
    for(int i=0; i< num_gpus; i++)
    {
        magma_queue_create(&stream[i][0]);
    }
    

    LDA = ((N+31)/32)*32;
    matsize = N*LDA;
    vecsize = N*incx;
    nb = 32;
    //nb = 64;

    printf("block size = %d\n", (int) nb);
   
    TESTING_MALLOC_CPU( A,       magmaFloatComplex, matsize );
    TESTING_MALLOC_CPU( X,       magmaFloatComplex, vecsize );
    TESTING_MALLOC_CPU( Ycublas, magmaFloatComplex, vecsize );
    TESTING_MALLOC_CPU( Ymagma,  magmaFloatComplex, vecsize );
    for(i=0; i<num_gpus; i++)
    {     
        TESTING_MALLOC_CPU( Y[i], magmaFloatComplex, vecsize );
    }

    magma_setdevice(0);
    TESTING_MALLOC_DEV( dA,       magmaFloatComplex, matsize );
    TESTING_MALLOC_DEV( dYcublas, magmaFloatComplex, vecsize );

    for(i=0; i<num_gpus; i++)
    {      
        n_local[i] = ((N/nb)/num_gpus)*nb;
        if (i < (N/nb)%num_gpus)
            n_local[i] += nb;
        else if (i == (N/nb)%num_gpus)
            n_local[i] += N%nb;
        
        magma_setdevice(i);
        
        TESTING_MALLOC_DEV( d_lA[i], magmaFloatComplex, LDA*n_local[i] );// potentially bugged 
        TESTING_MALLOC_DEV( dX[i],   magmaFloatComplex, vecsize );
        TESTING_MALLOC_DEV( dY[i],   magmaFloatComplex, vecsize );
        
        printf("device %2d n_local = %4d\n", (int) i, (int) n_local[i]); 
    }
    magma_setdevice(0);

      

    //////////////////////////////////////////////////////////////////////////

    /* Initialize the matrix */
    lapackf77_clarnv( &ione, ISEED, &matsize, A );
    magma_cmake_hermitian( N, A, LDA );

    blocks = N / nb + (N % nb != 0);
    lwork = LDA * (blocks + 1);
    TESTING_MALLOC_CPU( C_work, magmaFloatComplex, lwork );
    for(i=0; i<num_gpus; i++){
           magma_setdevice(i);  
           TESTING_MALLOC_DEV( dC_work[i], magmaFloatComplex, lwork );
           //fillZero(dC_work[i], lwork);
    }
      
     magma_setdevice(0);


    //////////////////////////////////////////////////////////////////////////
   
    fp = fopen ("results_chemv_mgpu.csv", "w") ;
    if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

    printf("CHEMV magmaFloatComplex precision\n\n");

    printf( "   n   CUBLAS,Gflop/s   MAGMABLAS,Gflop/s      \"error\"\n" 
            "==============================================================\n");
    fprintf(fp, "   n   CUBLAS,Gflop/s   MAGMABLAS,Gflop/s      \"error\"\n" 
            "==============================================================\n");


//    for( offset = 0; offset< N; offset ++ )
    
    for(int size = istart ; size <= N ; size += 128)
    {
    //    printf("offset = %d ", offset);
        m = size ;
    //    m = N;
        // lda = ((m+31)/32)*32;// 
        lda = LDA; 
        flops = FLOPS( (float)m ) / 1e6;

        printf(      "N %5d ", (int) m );
        fprintf( fp, "%5d, ", (int) m );

        vecsize = m * incx;
        lapackf77_clarnv( &ione, ISEED, &vecsize, X );
        lapackf77_clarnv( &ione, ISEED, &vecsize, Y[0] );

        /* =====================================================================
           Performs operation using CUDA-BLAS
           =================================================================== */
        magma_setdevice(0);
        magma_csetmatrix_1D_col_bcyclic(m, m, A, LDA, d_lA, lda, num_gpus, nb); 
        magma_setdevice(0);

    
    
    magma_csetmatrix( m, m, A, LDA, dA, lda );
        magma_csetvector( m, Y[0], incx, dYcublas, incx );
        
        for(i=0; i<num_gpus; i++){
            magma_setdevice(i);
            magma_csetvector( m, X, incx, dX[i], incx );
            magma_csetvector( m, Y[0], incx, dY[i], incx );


            blocks    = m / nb + (m % nb != 0);
            magma_csetmatrix( lda, blocks, C_work, LDA, dC_work[i], lda );
        }

        magma_setdevice(0);
        start = get_current_time();
        cublasChemv( uplo, m-offset, alpha, dA + offset + offset * lda, lda, dX[0] + offset, incx, beta, dYcublas + offset, incx );
         
        end = get_current_time();

        magma_cgetvector( m, dYcublas, incx, Ycublas, incx );
                
        
        cuda_perf = flops / GetTimerValue(start,end);
        printf(     "%11.2f", cuda_perf );
        fprintf(fp, "%11.2f,", cuda_perf );
       
        
        magma_setdevice(0);

        
        start = get_current_time();
        

        if(nb == 32)
       { 

        magmablas_chemv2_mgpu_32_offset( uplo, m, alpha, d_lA, lda, dX, incx, beta, dY, incx, 
                dC_work, lwork, num_gpus, nb, offset);
 
        }
        else // nb = 64
       { 

        magmablas_chemv2_mgpu_offset( uplo, m, alpha, d_lA, lda, dX, incx, beta, dY, incx, 
                dC_work, lwork, num_gpus, nb, offset);
 
        }
    
            
        for(i=1; i<num_gpus; i++)
        {
           magma_setdevice(i);
           cudaDeviceSynchronize();
        }
      
        end = get_current_time();
        magma_perf = flops / GetTimerValue(start,end); 
        printf(     "%11.2f", magma_perf );
        fprintf(fp, "%11.2f,", magma_perf );
       

        for(i=0; i<num_gpus; i++)
        {        
            magma_setdevice(i);
            magma_cgetvector( m, dY[i], incx, Y[i], incx );
        }
        magma_setdevice(0);

        
#ifdef validate        

        for( j= offset;j<m;j++)
        {
            for(i=1; i<num_gpus; i++)
            {

//            printf("Y[%d][%d] = %15.14f\n", i, j, Y[i][j].x);
#if defined(PRECISION_z) || defined(PRECISION_c)
            Y[0][j].x = Y[0][j].x + Y[i][j].x;
                        Y[0][j].y = Y[0][j].y + Y[i][j].y;
#else 
            Y[0][j] = Y[0][j] + Y[i][j];
            
#endif 

            }
        }

/*

#if defined(PRECISION_z) || defined(PRECISION_c)
        
        for( j=offset;j<m;j++)
        {
            if(Y[0][j].x != Ycublas[j].x)
            {
                     printf("Y-multi[%d] = %f, %f\n",  j, Y[0][j].x, Y[0][j].y );
                     printf("Ycublas[%d] = %f, %f\n",  j, Ycublas[j].x, Ycublas[j].y);
            }
        }

#else 

        for( j=offset;j<m;j++)
        {
            if(Y[0][j] != Ycublas[j])
            {
                     printf("Y-multi[%d] = %f\n",  j, Y[0][j] );
                     printf("Ycublas[%d] = %f\n",  j, Ycublas[j]);
            }
        }

#endif

*/        
        /* =====================================================================
           Computing the Difference Cublas VS Magma
           =================================================================== */
       
        magma_int_t nw = m - offset ;
        blasf77_caxpy( &nw, &c_neg_one, Y[0] + offset, &incx, Ycublas + offset, &incx);
        error = lapackf77_clange( "M", &nw, &ione, Ycublas + offset, &nw, work );
            
#if  0
        printf(      "\t\t %8.6e", error / m );
        fprintf( fp, "\t\t %8.6e", error / m );

        /*
         * Extra check with cblas vs magma
         */
        cblas_ccopy( m, Y, incx, Ycublas, incx );
        cblas_chemv( CblasColMajor, CblasLower, m, 
                     CBLAS_SADDR(alpha), A, LDA, X, incx, 
                     CBLAS_SADDR(beta), Ycublas, incx );
 
        blasf77_caxpy( &m, &c_neg_one, Ymagma, &incx, Ycublas, &incx);
        error = lapackf77_clange( "M", &m, &ione, Ycublas, &m, work );
#endif

        printf(      "\t\t %8.6e", error / m );
        fprintf( fp, "\t\t %8.6e", error / m );
 
#endif 
        printf("\n");        
        fprintf(fp, "\n");        
    }
    
    fclose( fp ) ; 

    /* Free Memory */
    TESTING_FREE_CPU( A );
    TESTING_FREE_CPU( X );
    TESTING_FREE_CPU( Ycublas );
    TESTING_FREE_CPU( Ymagma  );
    TESTING_FREE_CPU( C_work  );

    magma_setdevice(0);
    TESTING_FREE_DEV( dA );
    TESTING_FREE_DEV( dYcublas );
    
    for(i=0; i<num_gpus; i++)
    { 
        TESTING_FREE_CPU( Y[i] );
        magma_setdevice(i);

        TESTING_FREE_DEV( d_lA[i]    );
        TESTING_FREE_DEV( dX[i]      );
        TESTING_FREE_DEV( dY[i]      );
        TESTING_FREE_DEV( dC_work[i] );
    }

    magma_setdevice(0);
 ///////////////////////////////////////////////////////////   
      

    /* Free device */
    TESTING_FINALIZE();
    return 0;
}        
