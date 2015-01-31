/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/

/* includes, system */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include <quark.h>

/* includes, project */
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* Flops formula */
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6. * FMULS_GETRF(m, n) + 2. * FADDS_GETRF(m, n) )
#else
#define FLOPS(m, n) (      FMULS_GETRF(m, n) +      FADDS_GETRF(m, n) )
#endif

double get_LU_error(magma_int_t M, magma_int_t N, 
            cuDoubleComplex *A, magma_int_t lda,
            cuDoubleComplex *LU, magma_int_t *IPIV)
{
  magma_int_t min_mn = min(M,N), intONE = 1, i, j;

  lapackf77_zlaswp( &N, A, &lda, &intONE, &min_mn, IPIV, &intONE);

  cuDoubleComplex *L = (cuDoubleComplex *) calloc (M*min_mn, sizeof(cuDoubleComplex));
  cuDoubleComplex *U = (cuDoubleComplex *) calloc (min_mn*N, sizeof(cuDoubleComplex));
  double  *work = (double *) calloc (M+1, sizeof(cuDoubleComplex));

  memset( L, 0, M*min_mn*sizeof(cuDoubleComplex) );
  memset( U, 0, min_mn*N*sizeof(cuDoubleComplex) );

  lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
  lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

  for(j=0; j<min_mn; j++)
    L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );

  double matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);
  cuDoubleComplex alpha = MAGMA_Z_ONE;
  cuDoubleComplex beta  = MAGMA_Z_ZERO;

  blasf77_zgemm("N", "N", &M, &N, &min_mn,
                &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

  for( j = 0; j < N; j++ ) {
    for( i = 0; i < M; i++ ) {
      MAGMA_Z_OP_NEG( LU[i+j*lda], LU[i+j*lda], A[i+j*lda]);
    }
  }
  double residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

  free(L);
  free(work);

  return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_mc
*/
int main( int argc, char** argv) 
{
    cuDoubleComplex *h_A, *h_A2;
    magma_int_t *ipiv;
    double flops, gpu_perf, cpu_perf, cpu2_perf;

    magma_timestr_t start, end;

    /* Matrix size */
    magma_int_t N=0, n2, lda, M=0;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    magma_int_t i, j, info[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_int_t num_cores = 4;
    magma_int_t num_gpus  = 0;

    magma_int_t EN_BEE = -1;

    magma_int_t loop = argc;

    if (argc != 1){
      for(i = 1; i<argc; i++){      
        if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
        else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-C", argv[i])==0)
          num_cores = atoi(argv[++i]);
        else if (strcmp("-b", argv[i])==0)
          EN_BEE = atoi(argv[++i]);
      }
      if ((M>0 && N>0) || (M==0 && N==0)) {
        printf("  testing_zgetrf_mc -M %d -N %d -b %d -C %d\n\n", 
           M, N, EN_BEE, num_cores);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } else {
        printf("\nUsage: \n");
    printf("  Make sure you set the number of BLAS threads to 1, e.g.,\n");
    printf("   > setenv MKL_NUM_THREADS 1\n");
        printf("   > testing_zgetrf_mc -M %d -N %d -C 4 -b 128\n\n", 1024, 1024);
        exit(1);
      }
    } else {
      printf("\nUsage: \n");
      printf("  Make sure you set the number of BLAS threads to 1, e.g.,\n");
      printf("   > setenv MKL_NUM_THREADS 1\n");
      printf("   > testing_zgetrf_mc -M %d -N %d -C 4 -b 128\n\n", 1024, 1024);
      M = N = size[9];
    }

    n2 = M * N;
    magma_int_t min_mn = min(M, N);
    
    /* Allocate host memory for the matrix */
    TESTING_MALLOC( h_A2, cuDoubleComplex, n2    );
    TESTING_MALLOC( h_A , cuDoubleComplex, n2    );
    TESTING_MALLOC( ipiv, magma_int_t    , min_mn);

    /* Initialize MAGMA hardware context, seeting how many CPU cores 
       and how many GPUs to be used in the consequent computations  */
    magma_context *context;
    context = magma_init(NULL, NULL, 0, num_cores, num_gpus, argc, argv);

    printf("\n\n");
    printf("  M    N           GFlop/s        ||PA-LU|| / (||A||*N)\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){

      if (loop == 1) {
        M = N = min_mn = size[i];
        n2 = M*N;
      }

      flops = FLOPS( (double)M, (double)N ) / 1000000;

      /* Initialize the matrix */
      lapackf77_zlarnv( &ione, ISEED, &n2, h_A2 );
      lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A2, &M, h_A, &M );

      /* =====================================================================
         Performs operation using multi-core
         =================================================================== */

      start = get_current_time();
      magma_zgetrf_mc(context, &M, &N, h_A2, &M, ipiv, info);
      end = get_current_time();

      if (info[0] < 0)      
        printf("Argument %d of magma_sgeqrf_mc had an illegal value.\n", -info[0]);

      cpu2_perf = flops / GetTimerValue(start, end);
  
      double error = get_LU_error(M, N, h_A, M, h_A2, ipiv);

      printf("%5d %5d       %6.2f                  %e\n",
             M, N, cpu2_perf, error);

      if (loop != 1)
        break;
    }

    /* Memory clean up */
    TESTING_FREE( h_A2 );
    TESTING_FREE( h_A  );
    TESTING_FREE( ipiv );

    /* Shut down the MAGMA context */
    magma_finalize(context);
}
