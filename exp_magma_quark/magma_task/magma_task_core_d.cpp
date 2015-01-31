/* 
    -- MAGMA (version 1.6.1) -- 
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

#include "core_d.h"

#include "magma_task_core_d.h"

#if defined(MAGMA_WITH_MKL)
#include "mkl_service.h"
#endif
/*extern "C" int CORE_zgetrf_reclap(int M, int N,
                       double *A, int LDA,
                       int *IPIV, int *info);*/

/*important, just to create sometimes a fake dependencies*/
void magma_task_core_void(Schedule* sched_obj)
{    
    /*nothing*/
}

/*CPU wrapper*/
/*Fill block of memory: sets the first n bytes of the block of memory pointed by ptr to the specified value*/
 void magma_task_core_dmemset(Schedule* sched_obj )
 {
    double *ptr;
    double value;
    magma_int_t n;
    int i;

    schedule_unpack_args_3(sched_obj, ptr, value,  n);

    for(i=0;i<n;i++) ptr[i] = value;
 }

void magma_task_core_dgetrf(Schedule* sched_obj )
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

    //mkl_set_num_threads_local(16);

    schedule_unpack_args_6(sched_obj,m,  n, A,  LDA,  ipiv,  info);

    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(Before dgetrf)");
    #endif

    //magma_setlapack_numthreads(4); //testing
    

    dgetrf(&m,  &n, A,  &LDA,  ipiv,  info);

    //magma_setlapack_numthreads(1); //to be sure
    
     //magma_task_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, &ipiv[0], iinfo);
    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(after dgetrf)");
    #endif

    //mkl_set_num_threads_local(1);
#if (dbglevel >=1)
ca_trace_end_cpu('P');
#endif
}
/*
void magma_task_core_dgetrf_rec(Schedule* sched_obj )
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

     //magma_task_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, &ipiv[0], iinfo);
    #if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(after dgetrf)");
    #endif
}
*/
void magma_task_core_zgetrf_reclap(Schedule* sched_obj)
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

    //mkl_set_num_threads_local(1);
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

void magma_task_core_dtslu(Schedule* sched_obj)
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

    int info;
    int numthreads;
    int thread_num;
#if (dbglevel >=1)
    ca_trace_start();
#endif

    //mkl_set_num_threads_local(1);
//    printf("doing dgetrf\n");
    schedule_unpack_args_7(sched_obj, m, n, A, LDA, IPIV, iinfo, numthreads );

    
#if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(Before core_dtslu)");
#endif

    //printf("doing panel m:%d\n",M);
    thread_num = QUARK_Get_RankInTask(sched_obj); 
    

   core_dtslu( m, n, A, LDA, IPIV, &info, numthreads, thread_num);

   

    if(thread_num==0 && info!=0)
       printf("core_dtslu returned info:%d\n",info);

    iinfo += info; 

#if (dbglevel==10)
            ca_dbg_printMat(m, n, A, LDA, "A(after core_dtslu)");
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

 void magma_task_core_dlaswp(Schedule* sched_obj )
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
    //mkl_set_num_threads_local(1);

    schedule_unpack_args_7(sched_obj,n, A,  LDA,  K1,  K2,  ipiv,  incx);
    dlaswp( &n, A,  &LDA,  &K1,  &K2,  ipiv,  &incx);
      //magma_task_core_dlaswp(ncols, A(K,J), A_LD, c_one, nb, &ipiv[K], c_one);
 #if (dbglevel >=1)
ca_trace_end_cpu('W');
#endif
 }
     
void  magma_task_core_dtrsm(Schedule* sched_obj )
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
    //mkl_set_num_threads_local(1);

    schedule_unpack_args_11(sched_obj,side,  uplo,  transa,  diag, m,  n, alpha, A,  LDA, B,  LDB);

#if (dbglevel==10)
            if(side=='L')
                ca_dbg_printMat(m, n, B, LDB, "A[K,J](Before dtrsm)");
            else
                ca_dbg_printMat(m, n, B, LDB, "A[I,K](Before dtrsm)");
#endif

    dtrsm( &side,  &uplo,  &transa,  &diag, &m,  &n, &alpha, A,  &LDA, B,  &LDB);
    //magma_task_core_dtrsm("L", "L", "N", "U", nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD);
#if (dbglevel==10)
            if(side=='L')
                ca_dbg_printMat(m, n, B, LDB, "A[K,J](After dtrsm)");
            else
                ca_dbg_printMat(m, n, B, LDB, "A[I,K](After dtrsm)");
#endif
#if (dbglevel >=1)
if(side=='L'){
    ca_trace_end_cpu('U');
}
else{
    ca_trace_end_cpu('L');
}

#endif
}

void magma_task_core_dgemm(Schedule* sched_obj )
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
    //mkl_set_num_threads_local(1);

    schedule_unpack_args_13(sched_obj, transa,  transb, m,  n,  k, alpha, A,  LDA, B,  LDB, beta, C,  LDC);
    #if (dbglevel==10)
            ca_dbg_printMat(m, k, A, LDA, "A[I,K](Before dgemm)");
            ca_dbg_printMat(k, n, B, LDB, "A[K,J](Before dgemm)");
            ca_dbg_printMat(m, n, C, LDC, "A[I,J](Before dgemm)");
    #endif
    dgemm( &transa,  &transb, &m,  &n,  &k, &alpha, A,  &LDA, B,  &LDB, &beta, C,  &LDC);
     //magma_task_core_dgemm("N","N", nrows, ncols, nb, c_neg_one, A(I,K), A_LD, A(K,J), A_LD, c_one, A(I,J), A_LD);
    #if (dbglevel==10)
            ca_dbg_printMat(m, k, A, LDA, "A[I,K](After dgemm)");
            ca_dbg_printMat(k, n, B, LDB, "A[K,J](After dgemm)");
            ca_dbg_printMat(m, n, C, LDC, "A[I,J](After dgemm)");
    #endif
#if (dbglevel >=1)
ca_trace_end_cpu('S');
#endif
}

