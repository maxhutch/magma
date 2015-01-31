/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/

/* Includes, system */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include <quark.h>

/* Includes, project */
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6.*FMULS_GEQRF(m, n) + 2.*FADDS_GEQRF(m, n) )
#else
#define FLOPS(m, n) (    FMULS_GEQRF(m, n) +    FADDS_GEQRF(m, n) )
#endif

#include <pthread.h>

#define PRECISION_z


/* ------------------------------------------------------------
 * MAGMA QR params
 * --------------------------------------------------------- */
typedef struct {

  /* Whether or not to restore upper part of matrix */
  magma_int_t flag;

  /* Number of MAGMA threads */
  magma_int_t nthreads;

  /* Block size for left side of matrix */
  magma_int_t nb;

  /* Block size for right side of matrix */
  magma_int_t ob;

  /* Block size for final factorization */
  magma_int_t fb;

  /* Block size for multi-core factorization */
  magma_int_t ib;

  /* Number of panels for left side of matrix */
  magma_int_t np_gpu;

  /* Number of rows */
  magma_int_t m;

  /* Number of columns */
  magma_int_t n;

  /* Leading dimension */
  magma_int_t lda;

  /* Matrix to be factorized */
  cuDoubleComplex *a;

  /* Storage for every T */
  cuDoubleComplex *t;

  /* Flags to wake up MAGMA threads */
  volatile cuDoubleComplex **p;

  /* Synchronization flag */
  volatile magma_int_t sync0;

  /* One synchronization flag for each MAGMA thread */
  volatile magma_int_t *sync1;
  
  /* Synchronization flag */
  volatile magma_int_t sync2;

  /* Work space */
  cuDoubleComplex *w;

} magma_qr_params;

//magma_qr_params MG;

typedef struct {
  magma_int_t tid;
  void *params;
} t_params;


/* Update thread */
extern "C" void *cpu_thread(void *a)
{
  magma_int_t i;
  t_params *tp = (t_params*)a;

  magma_qr_params *mp = (magma_qr_params*)tp->params;

  //long magma_int_t t = (long int) a;
  long int t = (long int) tp->tid;

  magma_int_t M;
  magma_int_t N;
  magma_int_t K;
  cuDoubleComplex *WORK;

loop:

  while (mp->sync0 == 0) {
    sched_yield();
  }
    
  for (i = 0; i < mp->np_gpu; i++) 
  {

    while (mp->p[i] == NULL) {
      sched_yield();
    }
    
    M=mp->m-i*mp->nb;
    N=mp->ob;
    K=mp->nb;
    if (mp->m >= (mp->n-(mp->nthreads*mp->ob))) {
      if (i == (mp->np_gpu - 1)) {
        K = mp->n-mp->nthreads*mp->ob-(mp->np_gpu-1)*mp->nb; 
      }
    }

    WORK = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*M*N);
      
    lapackf77_zlarfb(MagmaLeftStr, MagmaTransStr, MagmaForwardStr, MagmaColumnwiseStr,
              &M,&N,&K,mp->a+i*mp->nb*mp->lda+i*mp->nb,&(mp->lda),mp->t+i*mp->nb*mp->nb,&K,
              mp->a+mp->m*mp->n-(mp->nthreads-t)*mp->ob*mp->lda+i*mp->nb,&(mp->lda),WORK,&N);
      
    free(WORK);
  }

  mp->sync1[t] = 1;
  
  while (mp->sync2 == 0) {
    sched_yield();
  }

goto loop;
    
  return (void*)NULL;
}

void magma_qr_init(magma_qr_params *qr_params,
                   magma_int_t m, magma_int_t n, cuDoubleComplex *a, magma_int_t nthreads)
{
  magma_int_t i;

  qr_params->nthreads = nthreads;

  if (qr_params->nb == -1)
    qr_params->nb = 128;

  if (qr_params->ob == -1)
    qr_params->ob = qr_params->nb;

  if (qr_params->fb == -1)
    qr_params->fb = qr_params->nb;

  if (qr_params->ob * qr_params->nthreads >= n){
    fprintf(stderr,"\n\nNumber of threads times block size not less than width of matrix.\n\n");
    exit(1);
  }

  qr_params->np_gpu = (n-(qr_params->nthreads * qr_params->ob)) / qr_params->nb;

  if ( (n-(qr_params->nthreads * qr_params->ob)) % qr_params->nb != 0)
    qr_params->np_gpu++;

  qr_params->m = m;
  qr_params->n = n;
  qr_params->lda = m;
  qr_params->a = a;
  qr_params->t = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*
                  qr_params->n*
                  qr_params->nb);

  if ((qr_params->n-(qr_params->nthreads*qr_params->ob)) > qr_params->m) {
    qr_params->np_gpu = m/qr_params->nb;
    if (m%qr_params->nb != 0)
      qr_params->np_gpu++;
  }

  fprintf(stderr,"qr_params->np_gpu=%d\n",qr_params->np_gpu);

  qr_params->p = (volatile cuDoubleComplex **) malloc (sizeof(cuDoubleComplex*)*
                  qr_params->np_gpu);

  for (i = 0; i < qr_params->np_gpu; i++)
    qr_params->p[i] = NULL;

  qr_params->sync0 = 1;
  
  qr_params->w = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*
                 qr_params->np_gpu*qr_params->nb*
                 qr_params->nb);

  qr_params->flag = 0;

  qr_params->sync2 = 0;

}

int TRACE;


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( magma_int_t argc, char** argv) 
{
    magma_int_t nquarkthreads=2;
    magma_int_t nthreads=2;
    magma_int_t num_gpus  = 1;
    TRACE = 0;

    //magma_qr_params mp;

    cuDoubleComplex *h_A, *h_R, *h_work, *tau;
    double gpu_perf, cpu_perf, flops;

    magma_timestr_t start, end;

    magma_qr_params *mp = (magma_qr_params*)malloc(sizeof(magma_qr_params));

    /* Matrix size */
    magma_int_t M=0, N=0, n2;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    cublasStatus status;
    magma_int_t i, j, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    mp->nb=-1;
    mp->ob=-1;
    mp->fb=-1;
    mp->ib=32;

    magma_int_t loop = argc;
    magma_int_t accuracyflag = 1;

    char precision;

    magma_int_t nc = -1;
    magma_int_t ncps = -1;

    if (argc != 1)
      {
    for(i = 1; i<argc; i++){      
      if (strcmp("-N", argv[i])==0)
        N = atoi(argv[++i]);
      else if (strcmp("-M", argv[i])==0)
        M = atoi(argv[++i]);
      else if (strcmp("-F", argv[i])==0)
        mp->fb = atoi(argv[++i]);
      else if (strcmp("-O", argv[i])==0)
        mp->ob = atoi(argv[++i]);
      else if (strcmp("-B", argv[i])==0)
        mp->nb = atoi(argv[++i]);
      else if (strcmp("-b", argv[i])==0)
        mp->ib = atoi(argv[++i]);
      else if (strcmp("-A", argv[i])==0)
        accuracyflag = atoi(argv[++i]);
      else if (strcmp("-P", argv[i])==0)
        nthreads = atoi(argv[++i]);
      else if (strcmp("-Q", argv[i])==0)
        nquarkthreads = atoi(argv[++i]);
      else if (strcmp("-nc", argv[i])==0)
        nc = atoi(argv[++i]);
      else if (strcmp("-ncps", argv[i])==0)
        ncps = atoi(argv[++i]);
    }
    
    if ((M>0 && N>0) || (M==0 && N==0)) 
      {
        printf("  testing_zgeqrf-v2 -M %d -N %d\n\n", M, N);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } 
    else 
      {
        printf("\nUsage: \n");
        printf("  Make sure you set the number of BLAS threads to 1, e.g.,\n");
        printf("   > setenv MKL_NUM_THREADS 1\n");
        printf("   > testing_zgeqrf-v2 -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
        exit(1);
      }
      } 
    else 
      {
    printf("\nUsage: \n");
    printf("  Make sure you set the number of BLAS threads to 1, e.g.,\n");
        printf("   > setenv MKL_NUM_THREADS 1\n");
        printf("  Set number of cores per socket and number of cores.\n");
    printf("   > testing_zgeqrf-v2 -M %d -N %d -ncps 6 -nc 12\n\n", 1024, 1024);
        printf("  Alternatively, set:\n");
        printf("  Q:  Number of threads for panel factorization.\n");
        printf("  P:  Number of threads for trailing matrix update (CPU).\n");
        printf("  B:  Block size.\n");
        printf("  b:  Inner block size.\n");
        printf("  O:  Block size for trailing matrix update (CPU).\n");
    printf("   > testing_zgeqrf-v2 -M %d -N %d -Q 4 -P 4 -B 128 -b 32 -O 200\n\n", 10112, 10112);
    M = N = size[9];
      }

    /* Auto tune based on number of cores and number of cores per socket if provided */
    if ((nc > 0) && (ncps > 0)) {
      precision = 's';
      #if (defined(PRECISION_d))
        precision = 'd';
      #endif
      #if (defined(PRECISION_c))
        precision = 'c';
      #endif
      #if (defined(PRECISION_z))
        precision = 'z';
      #endif
            
      auto_tune('q', precision, nc, ncps, M, N,
                &(mp->nb), &(mp->ob), &(mp->ib), &nthreads, &nquarkthreads);
          
fprintf(stderr,"%d %d %d %d %d\n",mp->nb,mp->ob,mp->ib,nquarkthreads,nthreads);
          
    }       

    /* Initialize MAGMA hardware context, seeting how many CPU cores
       and how many GPUs to be used in the consequent computations  */
    mp->sync0 = 0;
    magma_context *context;
    context = magma_init((void*)(mp),cpu_thread, nthreads, nquarkthreads, num_gpus, argc, argv);
    context->params = (void *)(mp);

    mp->sync1 = (volatile magma_int_t *) malloc (sizeof(int)*nthreads);

    for (i = 0; i < nthreads; i++)
      mp->sync1[i] = 0;

    n2  = M * N;
    magma_int_t min_mn = min(M, N);
    magma_int_t nb = magma_get_zgeqrf_nb(min_mn);
    magma_int_t lwork = N*nb;

    /* Allocate host memory for the matrix */
    TESTING_MALLOC   ( h_A  , cuDoubleComplex, n2    );
    TESTING_MALLOC   ( tau  , cuDoubleComplex, min_mn);
    TESTING_HOSTALLOC( h_R  , cuDoubleComplex, n2    );
    TESTING_HOSTALLOC(h_work, cuDoubleComplex, lwork );

    printf("\n\n");
    printf("  M     N   CPU GFlop/s   GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("==========================================================\n");
    for(i=0; i<10; i++){
        if (loop==1){
            M = N = min_mn = size[i];
            n2 = M*N;
        }

        flops = FLOPS( (double)M, (double)N ) / 1000000;

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &M, h_R, &M );

        //magma_zgeqrf(M, N, h_R, M, tau, h_work, lwork, &info);

        for(j=0; j<n2; j++)
          h_R[j] = h_A[j];

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_qr_init(mp, M, N, h_R, nthreads);

        start = get_current_time();
        magma_zgeqrf3(context, M, N, h_R, M, tau, h_work, lwork, &info);
        end = get_current_time();

        gpu_perf = flops / GetTimerValue(start, end);

    /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        if (accuracyflag == 1)
          lapackf77_zgeqrf(&M, &N, h_A, &M, tau, h_work, &lwork, &info);
        end = get_current_time();
        if (info < 0)
      printf("Argument %d of zgeqrf had an illegal value.\n", -info);

        cpu_perf = 4.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
    
        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        double work[1], matnorm = 1.;
        cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
        magma_int_t one = 1;

        if (accuracyflag == 1){
          matnorm = lapackf77_zlange("f", &M, &N, h_A, &M, work);
          blasf77_zaxpy(&n2, &mone, h_A, &one, h_R, &one);
        }

        if (accuracyflag == 1){
          printf("%5d %5d  %6.2f         %6.2f        %e\n",
                 M, N, cpu_perf, gpu_perf,
                 lapackf77_zlange("f", &M, &N, h_R, &M, work) / matnorm);
        } else {
          printf("%5d %5d                %6.2f          \n",
                 M, N, gpu_perf);
        }

        if (loop != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE    ( h_A  );
    TESTING_FREE    ( tau  );
    TESTING_HOSTFREE(h_work);
    TESTING_HOSTFREE( h_R  );

    /* Shut down the MAGMA context */
    magma_finalize(context);
}
