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
#include <quark.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_POTRF(n) + 2. * FADDS_POTRF(n) )
#else
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf_mc
*/
int main( magma_int_t argc, char** argv) 
{
    cuDoubleComplex *h_A, *h_R, *h_work, *h_A2;
    cuDoubleComplex *d_A;
    float gpu_perf, cpu_perf, cpu_perf2;

    magma_timestr_t start, end;

    /* Matrix size */
    magma_int_t N=0, n2, lda;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6048,7200,8064,8928,10080};
    
    magma_int_t i, j, info[1];

    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_int_t num_cores = 4;
    int num_gpus = 0;

    magma_int_t loop = argc;
    
    if (argc != 1){
      for(i = 1; i<argc; i++){      
        if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
        else if (strcmp("-C", argv[i])==0)
          num_cores = atoi(argv[++i]);
      }
      if (N==0) {
        N = size[9];
        loop = 1;
      } else {
        size[0] = size[9] = N;
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_zpotrf_mc -N %d -B 128 \n\n", 1024);
      N = size[9];
    }

    lda = N;
    n2 = size[9] * size[9];

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    /* Allocate host memory for the matrix */
    h_A2 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A2)\n");
    }

    /* Initialize MAGMA hardware context, seeting how many CPU cores 
       and how many GPUs to be used in the consequent computations  */
    magma_context *context;
    context = magma_init(NULL, NULL, 0, num_cores, num_gpus, argc, argv);

    
    printf("\n\n");
    printf("  N    Multicore GFlop/s    ||R||_F / ||A||_F\n");
    printf("=============================================\n");
    for(i=0; i<10; i++)
      {
    N = lda = size[i];
    n2 = N*N;

    lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
    
    for(j=0; j<N; j++) 
      MAGMA_Z_SET2REAL( h_A[j*lda+j], ( MAGMA_Z_GET_X(h_A[j*lda+j]) + 2000. ) );

    for(j=0; j<n2; j++)
      h_A2[j] = h_A[j];

    /* =====================================================================
       Performs operation using LAPACK 
       =================================================================== */

    //lapackf77_zpotrf("L", &N, h_A, &lda, info);
    lapackf77_zpotrf("U", &N, h_A, &lda, info);
    
    if (info[0] < 0)  
      printf("Argument %d of zpotrf had an illegal value.\n", -info[0]);     

    /* =====================================================================
       Performs operation using multi-core 
       =================================================================== */
    start = get_current_time();
    //magma_zpotrf_mc(context, "L", &N, h_A2, &lda, info);
    magma_zpotrf_mc(context, "U", &N, h_A2, &lda, info);
    end = get_current_time();
    
    if (info[0] < 0)  
      printf("Argument %d of magma_zpotrf_mc had an illegal value.\n", -info[0]);     
  
    cpu_perf2 = FLOPS( (double)N ) / (1000000.*GetTimerValue(start,end));
    
    /* =====================================================================
       Check the result compared to LAPACK
       =================================================================== */
    double work[1], matnorm = 1.;
    cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
    int one = 1;

    matnorm = lapackf77_zlange("f", &N, &N, h_A, &N, work);
    blasf77_zaxpy(&n2, &mone, h_A, &one, h_A2, &one);
    printf("%5d     %6.2f                %e\n", 
           size[i], cpu_perf2,  
           lapackf77_zlange("f", &N, &N, h_A2, &N, work) / matnorm);

    if (loop != 1)
      break;
      }
    
    /* Memory clean up */
    free(h_A);
    free(h_A2);

    /* Shut down the MAGMA context */
    magma_finalize(context);


}
