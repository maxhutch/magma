/*

    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 */
/*
 * Enable some functions for debug
 * dbglevel = 1: activate this file
 *
 * dbglevel = 10: print the matrix after each step
 * dbgleading=X , print the leading principal part of the matrix (useful for debug).
 */
#include <math.h>
#if defined( _WIN32 )
#  include <time.h>
#  include <sys/timeb.h>
#else
#  include <sys/time.h>
#endif

#ifdef USE_CUDBG
#include <windows.h>
#include "cudbg.h"    /*cuda debug mode*/
#include "cudbg_magma_wrap.h" 
#else
#include "common_magma.h"
#endif

#ifdef LLOG
#include "Llog.h"
#endif

static pthread_mutex_t mutex_print = PTHREAD_MUTEX_INITIALIZER; /*NOTE: For Debug*/

#ifdef LLOG
static Llog **llogs;
static int _P;
static int _ngpu;
static long tStart;
static long tEnd;
#endif




#ifdef _WIN32 
int ca_dbg_gettimeofday (struct timeval *tp, void *tz)
 {
struct _timeb timebuffer;
_ftime (&timebuffer);
 tp->tv_sec = (long) timebuffer.time;
 tp->tv_usec = timebuffer.millitm * 1000;
 return 0;
 }
#endif

long ca_dbg_usecs (){
  struct timeval t;
#ifdef _WIN32
  ca_dbg_gettimeofday(&t,NULL);
#else
  gettimeofday(&t,NULL);
#endif
  return t.tv_sec*1000000+t.tv_usec;
}

/*init*/
void ca_dbg_trace_init(int P, int ngpu)
{
#ifdef LLOG
    int i;
    tStart = ca_dbg_usecs ();
    _P = P;
    _ngpu = ngpu;
    llogs = (Llog**) malloc((_P+_ngpu) * sizeof(Llog*));
    for(i=0;i<_P+_ngpu;i++)
    {
        llogs[i]=NULL;
    }
#endif
}



/*trace section*/
void ca_dbg_trace_add_event(int tid, char type, long tStart, long tEnd, int step, int col, int row, int stolen)
{
#ifdef LLOG
    Llog_add_tail(&llogs[tid], tid, type, tStart, tEnd, step, col, row, stolen);
#endif
}

/*plot the trace and free memory*/
void ca_dbg_trace_finalize()
{
#ifdef LLOG
int i;
tEnd = ca_dbg_usecs ();

Llog_fplot(llogs, _P+_ngpu, tStart, tEnd, 1975, "Pthreads");//1875

for(i=0;i<_P+_ngpu;i++)
{
    Llog_free(&llogs[i]);
}
free(llogs);
#endif
}

/*get the current number of threads for the tracing*/
int ca_dbg_trace_get_P()
{
#ifdef LLOG
    return _P;
#endif
    return 0;
}

/*Synchronize on the device only when need*/
void ca_dbg_trace_device_sync()
{
#ifdef LLOG
    magma_device_sync();
#endif
}

/**/
/*print a vector of integer*/
void ca_dbg_iprintVec(int M, int *V, char desc[] )
{
int i, Mp;


    pthread_mutex_lock (&mutex_print);
         

#ifdef dbgleadingprint
        printf("Vector: %s M:%d  (Leading matrix: %d )\n",desc,M, dbgleadingprint);   
        Mp = min(dbgleadingprint,M); 
#else
        printf("Matrix: %s M:%d \n",desc,M);   
        Mp =M; 
#endif
        for(i=0;i<Mp;i++)
        {
            printf("%d ",V[i]);
        }
printf("\n");
    pthread_mutex_unlock (&mutex_print);
}

/*print a matrix*/
void ca_dbg_printMat(int M, int N, double *A,int LDA, char desc[] )
{
int i,j, Mp, Np;

    if ( magma_is_devptr( A ) == 1 ) {
        fprintf( stderr, "ERROR: printMat called with device pointer.\n" );
        exit(1);
    }

    pthread_mutex_lock (&mutex_print);
         

#ifdef dbgleadingprint
        printf("Matrix: %s M:%d N:%d (Leading matrix: %d x %d)\n",desc,M,N, dbgleadingprint, dbgleadingprint);   
        Mp = min(dbgleadingprint,M); Np =min(dbgleadingprint,N);
#else
        printf("Matrix: %s M:%d N:%d\n",desc,M,N);   
        Mp =M; Np =N;
#endif
        for(i=0;i<Mp;i++)
        {
            for(j=0;j<Np;j++)
            {
                if(A[j*LDA+i]>=0) printf(" ");

                printf("%.2f ",A[j*LDA+i]);
            }
        printf("\n");
        }
    pthread_mutex_unlock (&mutex_print);
}

/*print the transpose of a matrix, M: number of rows of A (not transposed), N:number of columns*/
void ca_dbg_printMat_transpose(int M, int N, double *A,int LDA, char desc[] )
{
int i,j, Mp, Np;;

    if ( magma_is_devptr( A ) == 1 ) {
        fprintf( stderr, "ERROR: printMat called with device pointer.\n" );
        exit(1);
    }

    pthread_mutex_lock (&mutex_print);
         //LDA = N;
    #ifdef dbgleadingprint
        printf("Matrix: %s M:%d N:%d (Leading matrix: %d x %d)\n",desc,M,N, dbgleadingprint, dbgleadingprint);   
        Mp = min(dbgleadingprint,M); Np =min(dbgleadingprint,N);
    #else
        printf("Matrix: %s M:%d N:%d\n",desc,M,N);   
        Mp =M; Np =N;
    #endif

        printf("Matrix Transposed: %s MT:%d NT:%d\n",desc,N,M);    
        for(j=0;j<Np;j++)
        {
            for(i=0;i<Mp;i++)
            {
                if(A[j*LDA+i]>=0) printf(" ");
                printf("%.2f ",A[j*LDA+i]);
            }
        printf("\n");
        }
    pthread_mutex_unlock (&mutex_print);
}

void ca_dbg_printMat_gpu(int M, int N, double *dA,int dA_LDA, char desc[] )
{

    double *A;
    int LDA=dA_LDA;
     
    if ( magma_is_devptr( dA ) == 0 ) {
        fprintf( stderr, "ERROR: printMat_gpu called with host pointer.\n" );
        exit(1);
    }

    /*temporary alloc cpu workspace*/
    //if(!(A =    (double *)    malloc(max(LDA,M)*N*sizeof(double)))) {printf("Memory allocation failed for A in ca_dbg_printMat_gpu"); exit(1);}
    //if(magma_malloc_pinned((void **) &A, max(LDA,M)*N)!=MAGMA_SUCCESS) {printf("Memory allocation failed for A in ca_dbg_printMat_gpu"); exit(1);}
   // if(magma_dmalloc_pinned(&A, max(LDA,M)*N)!=MAGMA_SUCCESS) {printf("Memory allocation failed for A in ca_dbg_printMat_gpu\n"); exit(1);}
    if(magma_dmalloc_cpu(&A, max(LDA,M)*N)!=MAGMA_SUCCESS) {printf("Memory allocation failed for A in ca_dbg_printMat_gpu\n"); exit(1);}
    magma_dgetmatrix(M, N, dA, dA_LDA, A, LDA);

    ca_dbg_printMat(M, N, A,LDA, desc);

    //magma_free_pinned(A);
    magma_free_cpu(A);
}

/*print the transpose of a matrix allocated on a device, M: number of rows of dA (not transposed), N:number of columns*/
void ca_dbg_printMat_transpose_gpu(int M, int N, double *dA,int dA_LDA, char desc[] )
{
//int i,j;
double *A;
int LDA=dA_LDA;

    //pthread_mutex_lock (&mutex_gpu_print);
     if ( magma_is_devptr( dA ) == 0 ) {
        fprintf( stderr, "ERROR: printMat_gpu called with host pointer.\n" );
        exit(1);
    }
    /*temporary alloc cpu workspace*/
    //printf("transpose_gpu M:%d, N:%d, LDA:%d\n",M,N, LDA);
    /*TODO slow*/
    //if (MAGMA_SUCCESS != magma_dmalloc_cpu(&A, max(LDA,M)*N)) {printf("Memory allocation failed for A in cudbg_dprint_transpose_gpu\n");exit(1);}
    //if(!(A =    (double *)    malloc(max(LDA,M)*N*sizeof(double)))) {printf("Memory allocation failed for A in cudbg_dprint_transpose_gpu\n");exit(1);}
    //if(magma_dmalloc_pinned(&A, max(LDA,M)*N)!=MAGMA_SUCCESS) {printf("Memory allocation failed for A in cudbg_dprint_transpose_gpu\n");exit(1);}
    if(magma_dmalloc_cpu(&A, max(LDA,M)*N)!=MAGMA_SUCCESS) {printf("Memory allocation failed for A in ca_dbg_printMat_gpu\n"); exit(1);}
    magma_dgetmatrix(M, N, dA, dA_LDA, A, LDA);
    
    ca_dbg_printMat_transpose(M, N, A,LDA, desc);

    //magma_free_pinned(A);
    //free(A);
    magma_free_cpu(A);
}

void ca_dbg_printMat_mgpu(int num_gpus, int M, int *N_local, double **dA,int dA_LDA, char desc[] ){

    int dd;
    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
        ca_dbg_printMat_gpu(M, N_local[dd], dA[dd], dA_LDA, desc); 
    }
}

/*print the transpose of a matrix allocated on a device, *M_local: number of rows of dA (not transposed), N:number of columns*/
void ca_dbg_printMat_transpose_mgpu(int num_gpus, int *M_local, int N, double **dA,int dA_LDA, char desc[] ){

    int dd;
    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
        ca_dbg_printMat_transpose_gpu( M_local[dd], N, dA[dd], dA_LDA, desc); 
    }
}

/*write a matrix in a file*/
void ca_dbg_fwriteMat(char *filename, int M, int N, double *A, int LDA)
{
    int i,j;
    FILE *f;
/*Write the matrix in a file*/
printf("Saving A in file...\n");
f = fopen(filename,"w+"); //data_12_1.txt //data_10_5.txt
if(f==NULL) {printf("File not found in ca_dbg_freadMat"); exit(1);}

for(j=0;j<N;j++)
 for(i=0;i<M;i++)
 {
    fprintf(f,"%f\n",A[LDA*i+j]);
 }
fclose(f);
}

/*read a matrix from a file*/
void ca_dbg_freadMat(char *filename, int M, int N, double *A, int LDA)
{
    int i,j;
    char buff[512];
    FILE *f;
    double val;
/*Read the matrix from a file*/
printf("Reading A in file...\n");
f = fopen(filename,"r"); //data_12_1.txt //data_10_5.txt
if(f==NULL) {printf("File not found in ca_dbg_freadMat"); exit(1);}

for(j=0;j<N;j++)
 for(i=0;i<M;i++)
 {
     fgets(buff,512,f);
     //fscanf(f,"%f",&RESID);
     val = atof(buff);
    A[LDA*i+j] = val;
 }
fclose(f);
}

