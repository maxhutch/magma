/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include <quark.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* Flops formula */
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6.*FMULS_GEQRF(m, n) + 2.*FADDS_GEQRF(m, n) )
#else
#define FLOPS(m, n) (    FMULS_GEQRF(m, n) +    FADDS_GEQRF(m, n) )
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( magma_int_t argc, char** argv) 
{
    cuDoubleComplex *h_A, *h_R, *h_A2, *h_A3, *h_work, *h_work2, *tau, *d_work2;
    cuDoubleComplex *d_A, *d_work;
    float gpu_perf, cpu_perf, cpu2_perf;
    double flops;

    magma_timestr_t start, end;

    /* Matrix size */
    magma_int_t N=0, n2, lda, M=0;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    magma_int_t i, j, info[1];

    magma_int_t loop = argc;

    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_int_t num_cores = 4;
    magma_int_t num_gpus = 0;

    if (argc != 1){
      for(i = 1; i<argc; i++){      
        if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
        else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-C", argv[i])==0)
         num_cores = atoi(argv[++i]);
      }
      if ((M>0 && N>0) || (M==0 && N==0)) {
        printf("  testing_zgeqrf_mc -M %d -N %d \n\n", M, N);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrf_mc -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
        exit(1);
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_zgeqrf_mc -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
      M = N = size[9];
    }

    n2 = M * N;

    magma_int_t min_mn = min(M,N);

    /* Allocate host memory for the matrix */
    h_A2 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A2)\n");
    }

    magma_int_t lwork = n2;

    h_work2 = (cuDoubleComplex*)malloc(lwork * sizeof(cuDoubleComplex));
    if (h_work2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_work2)\n");
    }

    h_A3 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A3[0]));
    if (h_A3 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A3)\n");
    }

    tau = (cuDoubleComplex*)malloc(min_mn * sizeof(cuDoubleComplex));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }

    /* Initialize MAGMA hardware context, seeting how many CPU cores 
       and how many GPUs to be used in the consequent computations  */
    magma_context *context;
    context = magma_init(NULL, NULL, 0, num_cores, num_gpus, argc, argv);

    printf("\n\n");
    printf("   M     N       LAPACK Gflop/s     Multi-core Gflop/s    ||R||_F / ||A||_F\n");
    printf("===========================================================================\n");
    for(i=0; i<10; i++){

      if (loop == 1) {
        M = N = size[i];
        n2 = M*N;
      }

      flops = FLOPS( (double)M, (double)N ) / 1000000;

      /* Initialize the matrix */
      lapackf77_zlarnv( &ione, ISEED, &n2, h_A2 );
      lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A2, &M, h_A3, &M );

      /* =====================================================================
         Performs operation using LAPACK 
     =================================================================== */

      start = get_current_time();
      lapackf77_zgeqrf(&M, &N, h_A3, &M, tau, h_work2, &lwork, info);
      end = get_current_time();

      if (info[0] < 0)  
        printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);
 
      cpu2_perf = flops / GetTimerValue(start, end);

      /* =====================================================================
         Performs operation using multicore 
     =================================================================== */

      start = get_current_time();
      magma_zgeqrf_mc(context, &M, &N, h_A2, &M, tau, h_work2, &lwork, info);
      end = get_current_time();

      if (info[0] < 0)  
        printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);
  
      cpu_perf = flops / GetTimerValue(start, end);
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */

      double work[1], matnorm = 1.;
      cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
      magma_int_t one = 1;
      matnorm = lapackf77_zlange("f", &M, &N, h_A2, &M, work);

      blasf77_zaxpy(&n2, &mone, h_A2, &one, h_A3, &one);
      printf("%5d  %5d       %6.2f               %6.2f           %e\n", 
         M,  N, cpu2_perf, cpu_perf,
         lapackf77_zlange("f", &M, &N, h_A3, &M, work) / matnorm);

      if (loop != 1)
    break;
    }

    /* Memory clean up */
    free(h_A2);
    free(tau);
    free(h_A3);
    free(h_work2);

    /* Shut down the MAGMA context */
    magma_finalize(context);

}
