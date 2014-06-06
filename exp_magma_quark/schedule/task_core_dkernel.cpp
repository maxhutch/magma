/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
#if (dbglevel >=1)
#include "ca_dbg_tools.h"
#endif



#include "common_magma.h"

#include "schedule.h"

#include "core_dkernel.h"

#include "task_core_dkernel.h"


/*extern "C" int CORE_zgetrf_reclap(int M, int N,
                       double *A, int LDA,
                       int *IPIV, int *info);*/

/*important, just to create sometimes a fake dependencies*/
void task_core_void(Schedule* sched_obj)
{    
    /*nothing*/
}

/*CPU wrapper*/
/*Fill block of memory: sets the first n bytes of the block of memory pointed by ptr to the specified value*/
 void task_core_dmemset(Schedule* sched_obj )
 {
    double *ptr;
    double value;
    magma_int_t n;
    int i;

    schedule_unpack_args_3(sched_obj, ptr, value,  n);

    for(i=0;i<n;i++) ptr[i] = value;
 }

void task_core_dgetrf(Schedule* sched_obj )
{
    magma_int_t m;
    magma_int_t n;
    double *A; 
    magma_int_t LDA; 
    magma_int_t *ipiv; 
    magma_int_t *info;

#if (dbglevel >=1)
ca_trace_start();
#endif

    schedule_unpack_args_6(sched_obj,m,  n, A,  LDA,  ipiv,  info);

    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(Before dgetrf)");
    #endif

    
    dgetrf(&m,  &n, A,  &LDA,  ipiv,  info);

     //task_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, &ipiv[0], iinfo);
    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(after dgetrf)");
    #endif

#if (dbglevel >=1)
ca_trace_end_cpu('P');
#endif
}
/*
void task_core_dgetrf_rec(Schedule* sched_obj )
{
    magma_int_t m;
    magma_int_t n;
    double *A; 
    magma_int_t LDA; 
    magma_int_t *ipiv; 
    magma_int_t *info;

    schedule_unpack_args_6(sched_obj,m,  n, A,  LDA,  ipiv,  info);

    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(Before dgetrf)");
    #endif

    dgetrf(&m,  &n, A,  &LDA,  ipiv,  info);

     //task_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, &ipiv[0], iinfo);
    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(after dgetrf)");
    #endif
}
*/
void task_core_zgetrf_reclap(Schedule* sched_obj)
{
    /*CORE_zgetrf_reclap_schedule(Schedule* sched_obj)*/
    int m;
    int n;
    double *A;
    int LDA;
    int *IPIV;
    /*
    PLASMA_sequence *sequence;
    PLASMA_request *request;
    PLASMA_bool check_info;
    */
    int iinfo;

    int info[3];
    int maxthreads;

#if (dbglevel >=1)
    ca_trace_start();
#endif

//    printf("doing dgetrf\n");
    schedule_unpack_args_7(sched_obj, m, n, A, LDA, IPIV, iinfo, maxthreads );

    
#if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(Before core_zgetrf_reclap)");
#endif

    //printf("doing panel m:%d\n",M);
    info[1] = QUARK_Get_RankInTask(sched_obj); 
    info[2] = maxthreads;

   CORE_zgetrf_reclap( m, n, A, LDA, IPIV, info );

   iinfo += info[0]; 

   if(info[1]==0 && info[0]!=0)
       printf("core_dgetrf returned info:%d\n",info[0]);

#if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(after core_zgetrf_reclap)");
#endif

    /*
    if (info[1] == 0 && info[0] != PLASMA_SUCCESS && check_info)
        plasma_sequence_flush(sched_obj, sequence, request, iinfo + info[0] );
    */
#if (dbglevel >=1)
ca_trace_end_cpu('P');
#endif
//    printf("end doing dgetrf\n");
}

 void task_core_dlaswp(Schedule* sched_obj )
 {
    magma_int_t n;
    double *A;
    magma_int_t LDA;
    magma_int_t K1;
    magma_int_t K2;
    magma_int_t *ipiv;
    magma_int_t incx;

#if (dbglevel >=1)
ca_trace_start();
#endif

    schedule_unpack_args_7(sched_obj,n, A,  LDA,  K1,  K2,  ipiv,  incx);
    dlaswp( &n, A,  &LDA,  &K1,  &K2,  ipiv,  &incx);
      //task_core_dlaswp(ncols, A(K,J), A_LD, c_one, nb, &ipiv[K], c_one);
 #if (dbglevel >=1)
ca_trace_end_cpu('W');
#endif
 }
     
void  task_core_dtrsm(Schedule* sched_obj )
{
    char side;
    char uplo;
    char transa;
    char diag; 
    magma_int_t m;
    magma_int_t n;
    double alpha; 
    double *A;
    magma_int_t LDA;
    double *B;
    magma_int_t LDB;

#if (dbglevel >=1)
ca_trace_start();
#endif
    schedule_unpack_args_11(sched_obj,side,  uplo,  transa,  diag, m,  n, alpha, A,  LDA, B,  LDB);

#if (dbglevel==10)
            ca_dbg_printMat(m, n, B, LDB, "A[K,J](Before dtrsm)");
#endif

    dtrsm( &side,  &uplo,  &transa,  &diag, &m,  &n, &alpha, A,  &LDA, B,  &LDB);
    //task_core_dtrsm("L", "L", "N", "U", nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD);
#if (dbglevel==10)
            ca_dbg_printMat(m, n, B, LDB, "A[K,J](after dtrsm)");
#endif
#if (dbglevel >=1)
ca_trace_end_cpu('U');
#endif
}

void task_core_dgemm(Schedule* sched_obj )
 {
    char transa;
    char transb; 
    magma_int_t m;
    magma_int_t n;
    magma_int_t k;
    double alpha; 
    double *A;
    magma_int_t LDA;
    double *B;
    magma_int_t LDB;
    double beta;
    double *C;
    magma_int_t LDC;

#if (dbglevel >=1)
ca_trace_start();
#endif

    schedule_unpack_args_13(sched_obj, transa,  transb, m,  n,  k, alpha, A,  LDA, B,  LDB, beta, C,  LDC);
    #if (dbglevel==10)
            ca_dbg_printMat(m, n, C, LDC, "A[I,J](Before dgemm)");
    #endif
    dgemm( &transa,  &transb, &m,  &n,  &k, &alpha, A,  &LDA, B,  &LDB, &beta, C,  &LDC);
     //task_core_dgemm("N","N", nrows, ncols, nb, c_neg_one, A(I,K), A_LD, A(K,J), A_LD, c_one, A(I,J), A_LD);
    #if (dbglevel==10)
            ca_dbg_printMat(m, n, C, LDC, "A[I,J](After dgemm)");
    #endif
#if (dbglevel >=1)
ca_trace_end_cpu('S');
#endif
}
