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
#include "magma_task_d.h"

#include "schedule.h"

//static pthread_mutex_t mutex_dAP = PTHREAD_MUTEX_INITIALIZER;

#define default_stream 0

void magma_task_dmalloc_pinned(Schedule* sched_obj )
 {
    magma_int_t size;
    double **A; 
    void *dep_ptr;

#if (dbglevel >=1)
    ca_trace_start();
#endif
//    printf("doing dmalloc\n");
    schedule_unpack_args_3(sched_obj, size, A, dep_ptr);
//    printf("doing dmalloc %p\n",dep_ptr);

    //printf("using malloc instead, *** TODO: fix\n");
    //A = (double**) malloc(size * sizeof(double));

    magma_dmalloc_pinned(A, size);
    
//    printf("end doing dmalloc\n");
#if (dbglevel >=1)
ca_trace_end_1gpu('O');
ca_trace_end_cpu('C');
#endif
 }

void magma_task_dfree_pinned(Schedule* sched_obj )
 {
    double *A; 
    void *dep_ptr;
#if (dbglevel >=1)
    ca_trace_start();
#endif
//    printf("doing dmalloc\n");
    schedule_unpack_args_2(sched_obj, A, dep_ptr);
//    printf("doing dmalloc %p\n",dep_ptr);

    //printf("using malloc instead, *** TODO: fix\n");
    //A = (double**) malloc(size * sizeof(double));

    magma_free_pinned(A);
    
#if (dbglevel >=1)
ca_trace_end_1gpu('O');
ca_trace_end_cpu('C');
#endif
 }

void magma_task_dfree_pinned_index(Schedule* sched_obj )
 {
    double **A; 
    magma_int_t index;
    void *dep_ptr;
#if (dbglevel >=1)
    ca_trace_start();
#endif
//    printf("doing dmalloc\n");
    schedule_unpack_args_3(sched_obj, A, index, dep_ptr);
//    printf("doing dmalloc %p\n",dep_ptr);

    //printf("using malloc instead, *** TODO: fix\n");
    //A = (double**) malloc(size * sizeof(double));

    //printf("*** using simpl free\n");
    //free(A[index]);

    //A[index]=NULL;
    magma_free_pinned(A[index]);
#if (dbglevel >=1)
ca_trace_end_1gpu('0');
ca_trace_end_cpu('C');
#endif

 }

void magma_task_dsetmatrix(Schedule* sched_obj )
 {
    magma_int_t m;
    magma_int_t nb;
    double *A_src;
    magma_int_t LDA;
    double *dA_dst;
    magma_int_t dA_LD;

    schedule_unpack_args_6(sched_obj, m,  nb, A_src,  LDA, dA_dst,  dA_LD);
    magma_dsetmatrix(m, nb, A_src, LDA, dA_dst, dA_LD);
      //task_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)
 }

 void magma_task_dgetmatrix(Schedule* sched_obj )               
 {
    magma_int_t m;
    magma_int_t nb;
    double *dA_src;
    magma_int_t dA_LD;
    double *A_dst;
    magma_int_t LDA;

    schedule_unpack_args_6(sched_obj, m,  nb, dA_src,  dA_LD, A_dst,  LDA);
    magma_dgetmatrix( m,  nb, dA_src,  dA_LD, A_dst,  LDA);
     // task_getpanel(gpu_nrows, ncols, dAT(K,K+A_N), dAT_LD, A(K,K+A_N), A_LD);
 }

 void magma_task_dsetmatrix_transpose(Schedule* sched_obj)
 {
    magma_int_t m;
    magma_int_t nb;
    double *A_src;
    magma_int_t LDA;
    double *dA_dst;
    magma_int_t dA_LD;
    double *dwork;
    magma_int_t dwork_LD;
#if (dbglevel >=1)
    ca_trace_start();
#endif

    schedule_unpack_args_8(sched_obj, m,  nb, A_src,  LDA, dA_dst,  dA_LD, dwork, dwork_LD);

    //magma_dsetmatrix_transpose( m,  nb, A_src,  LDA, dA_dst,  dA_LD);
      //task_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)

#if (dbglevel==10)
            ca_dbg_printMat(m, nb, A_src, LDA, "A before setMatrix");
#endif
//            pthread_mutex_lock(&mutex_dAP);
    /*1. copy A to dwork: send to the GPU*/
    magma_dsetmatrix(m, nb, A_src, LDA, dwork, dwork_LD);
#if (dbglevel==10)
//            ca_dbg_printMat_gpu(m, nb, dwork, dwork_LD, "dwork after setMatrix");
#endif
    /*2.transpose dwork to dA*/
    magmablas_dtranspose2(dA_dst, dA_LD, dwork, dwork_LD, m, nb);
//    pthread_mutex_unlock(&mutex_dAP);
#if (dbglevel==10)
            ca_dbg_printMat_transpose_gpu(nb, m, dA_dst, dA_LD, "dA after setMatrix");
#endif
#if (dbglevel >=1)
ca_trace_end_1gpu('T');
ca_trace_end_cpu('C');
#endif
 }

 void magma_task_dgetmatrix_transpose(Schedule* sched_obj )               
 {
    magma_int_t m;
    magma_int_t nb;
    double *dA_src;
    magma_int_t dA_LD;
    double *A_dst;
    magma_int_t LDA;
    double *dwork;
    magma_int_t dwork_LD;

    void *dep_ptr;

#if (dbglevel >=1)
    ca_trace_start();
#endif

//    printf("Matrix_get_transpose\n");
    //schedule_unpack_args_8(sched_obj, m,  nb, dA_src,  dA_LD, A_dst,  LDA, dwork, dwork_LD);
    schedule_unpack_args_9(sched_obj, m,  nb, dA_src,  dA_LD, A_dst,  LDA, dwork, dwork_LD, dep_ptr);

//    printf("Matrix_get_transpose m:%d, nb:%d, dep_ptr:%p\n",m,nb,dep_ptr);

    //magma_dgetmatrix_transpose( m,  nb, dA_src,  dA_LD, A_dst,  LDA);
     // task_getpanel(gpu_nrows, ncols, dAT(K,K+A_N), dAT_LD, A(K,K+A_N), A_LD);
#if (dbglevel==10)
            ca_dbg_printMat_transpose_gpu(nb, m, dA_src, dA_LD, "dA before getMatrix");
#endif
//            pthread_mutex_lock(&mutex_dAP);
    /*1.transpose dA to dwork*/
    magmablas_dtranspose2(dwork, dwork_LD, dA_src, dA_LD,  nb, m);
//    printf("Matrix_get_transpose m:%d, nb:%d 1:done\n",m,nb);
#if (dbglevel==10)
//            ca_dbg_printMat_gpu(m, nb, dwork, dwork_LD, "dwork after dA transpose");
#endif
    /*2. copy dwork to A: send the panel to GPU*/
    magma_dgetmatrix(m, nb, dwork, dwork_LD, A_dst, LDA);
//    printf("Matrix_get_transpose m:%d, nb:%d 2:done\n",m,nb);
//    pthread_mutex_unlock(&mutex_dAP);
#if (dbglevel==10)
            ca_dbg_printMat(m, nb, A_dst, LDA, "A after getMatrix");
#endif 
#if (dbglevel >=1)
ca_trace_end_1gpu('G');
ca_trace_end_cpu('C');
#endif

//    printf("End Matrix_get_transpose m:%d, nb:%d\n",m,nb);
 }

void magma_task_dlaswp(Schedule* sched_obj   )
{
    magma_int_t n;
    double *dA;
    magma_int_t lda;
    magma_int_t i1;
    magma_int_t i2;
    magma_int_t *ipiv;
    magma_int_t inci;
#if (dbglevel >=1)
    ca_trace_start();
#endif
    schedule_unpack_args_7(sched_obj,n, dA,  lda,  i1,  i2,  ipiv,  inci);


    magmablas_dlaswp(  n, dA,  lda,  i1,  i2,  ipiv,  inci );
     //magma_task_dlaswp(gpu_ncols, dAT(K,K+A_N), dAT_LD, c_one, nb, &ipiv[K], c_one);
#if (dbglevel >=1)
ca_trace_end_1gpu('W');
ca_trace_end_cpu('C');
#endif
}
/**/
void magma_task_dtrsm(Schedule* sched_obj  )
{
    magma_side_t side;
    magma_uplo_t uplo;
    magma_trans_t trans;
    magma_diag_t diag;
    magma_int_t m;
    magma_int_t n;
    double alpha;
    double *dA;
    magma_int_t lda;
    double *dB;
    magma_int_t ldb;
#if (dbglevel >=1)
    ca_trace_start();
#endif
    schedule_unpack_args_11(sched_obj,side,  uplo,  trans,  diag,  m,  n, alpha, dA,  lda, dB,  ldb);
    magma_dtrsm( side,  uplo,  trans,  diag,  m,  n, alpha, dA,  lda, dB,  ldb );
     //magma_task_dtrsm('R', 'U', 'N', 'U', gpu_ncols, nb, c_one, dAT(K,K), dAT_LD, dAT(K,K+A_N), dAT_LD);
#if (dbglevel >=1)
ca_trace_end_1gpu('U');
ca_trace_end_cpu('C');
#endif
}

void magma_task_dgemm(Schedule* sched_obj  )
{
    magma_trans_t transA;
    magma_trans_t transB;
    magma_int_t m;
    magma_int_t n;
    magma_int_t k;
    double alpha;
    double *dA;
    magma_int_t lda;
    double *dB;
    magma_int_t ldb;
    double beta;
    double *dC;
    magma_int_t ldc;

#if (dbglevel >=1)
    ca_trace_start();
#endif
    schedule_unpack_args_13(sched_obj,transA,  transB,  m,  n,  k, alpha, dA,  lda, dB,  ldb, beta, dC,  ldc);

#if (dbglevel==10)
            ca_dbg_printMat_transpose_gpu(m, k, dA, lda, "A before magma_dgemm");
            ca_dbg_printMat_transpose_gpu(k, n, dB, ldb, "B before magma_dgemm");
            ca_dbg_printMat_transpose_gpu(m, n, dC, ldc, "C before magma_dgemm");
#endif

    
    magma_dgemm( transA,  transB,  m,  n,  k, alpha, dA,  lda, dB,  ldb, beta, dC,  ldc );
     //magma_task_dgemm('N','N', gpu_ncols, gpu_nrows, nb, c_neg_one, dAT(K,K+A_N), dAT_LD, dAT(K+1,K), dAT_LD, c_one, dAT(K+1,K+A_N), dAT_LD);

#if (dbglevel==10)
            ca_dbg_printMat_transpose_gpu(m, n, dC, ldc, "C after magma_dgemm");
#endif
#if (dbglevel >=1)
ca_trace_end_1gpu('S');
ca_trace_end_cpu('C');
#endif
}


