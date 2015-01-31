/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
#include "common_magma.h"

#include "schedule.h"

//#include "task_core_d.h"
#include "magma_task_dev_d.h"




 /*CPU - GPU transfer wrapper*/
  void magma_insert_dev_dmalloc_pinned(magma_int_t deviceID, magma_int_t size, double **A, void *A_dep_ptr)
 {
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
    // schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;
     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dmalloc_pinned");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");
     
      schedule_Insert_Task(sched_obj, magma_task_dev_dmalloc_pinned, &task_flags,
            sizeof(magma_int_t),          &deviceID,VALUE,  
            sizeof(magma_int_t),          &size,       VALUE,
              ///sizeof(double**),                A,            OUTPUT,//*size
              sizeof(double**),                A,            OUTPUT,//*size
              //dependency to set after the allocation is completed
              sizeof(void *),              A_dep_ptr,    INOUT,
              
              0);
      
      //magma_dmalloc_pinned(A, size);
      
 }

 void magma_insert_dev_dfree_pinned(magma_int_t deviceID, double *A, void *A_dep_ptr)
 {
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
    // schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dfree_pinned");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");
     
      schedule_Insert_Task(sched_obj, magma_task_dev_dfree_pinned, &task_flags,
              ///sizeof(double**),                A,            OUTPUT,//*size
              sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(double*),                    A,        INOUT,//*size
              //dependency to set after the allocation is completed
              sizeof(void *),              A_dep_ptr,    INOUT,
              0);
      
      //magma_dmalloc_pinned(A, size);
      
 }

 void magma_insert_dev_dfree_pinned_index(magma_int_t deviceID, double **A, int index, void *A_dep_ptr)
 {
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
    // schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dfree_pinned_index");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");
     
      schedule_Insert_Task(sched_obj, magma_task_dev_dfree_pinned_index, &task_flags,
              ///sizeof(double**),                A,            OUTPUT,//*size
              sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(double**),                A,              NODEP,//*size
              sizeof(magma_int_t),          &index,       VALUE,
              //dependency to set after the allocation is completed
              sizeof(void *),              A_dep_ptr,    INOUT,
              0);
      
      //magma_dmalloc_pinned(A, size);
      
 }


  void magma_insert_dev_queue_sync(magma_int_t deviceID, magma_queue_t stream1, void *dep_ptr)
 {
     
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
   //  schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"queue_sync");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"black");

      schedule_Insert_Task(sched_obj, magma_task_dev_queue_sync, &task_flags,
              sizeof(magma_int_t),         &deviceID,    VALUE,
              sizeof(magma_queue_t),       &stream1,     VALUE,

              /*dependency to check before the synchronisation*/
              sizeof(void *),              dep_ptr,INOUT,
              /*dependency to set after the synchronisation*/
              //sizeof(void *),            output_dep_ptr,OUTPUT,
              0);

      //schedule_insert_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)
 }

  void magma_insert_dev_dsetmatrix(magma_int_t deviceID, magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD)
 {

     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_setmatrix");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");

      schedule_Insert_Task(sched_obj, magma_task_dev_dsetmatrix, &task_flags,
          sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*LDA*nb,           A_src,    INPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(double)*dA_LD*nb,         dA_dst,   OUTPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              0);

      //schedule_insert_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)
 }

 void magma_insert_dev_dgetmatrix(magma_int_t deviceID, magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA)               
 {
 
      Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

 schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
 //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
 //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
 schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

 schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_getmatrix");
 schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"indigo");

           schedule_Insert_Task(sched_obj, magma_task_dev_dgetmatrix, &task_flags,
               sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*dA_LD*nb,         dA_src,   INPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              sizeof(double)*LDA*nb,           A_dst,    OUTPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              0);
     // schedule_insert_getpanel(gpu_nrows, ncols, dAT(K,K+A_N), dAT_LD, A(K,K+A_N), A_LD);
 }

 void magma_insert_dev_dsetmatrix_transpose(magma_int_t deviceID, magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD, double *dwork, magma_int_t dwork_LD, void *A_src_dep_ptr, void *dA_dst_dep_ptr)
 {
     
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
   //  schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_setmatrix");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");

      schedule_Insert_Task(sched_obj, magma_task_dev_dsetmatrix_transpose, &task_flags,
          sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*LDA*nb,           A_src,    INPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(double)*dA_LD*nb,         dA_dst,   OUTPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              sizeof(double)*dwork_LD*nb,      dwork,    INOUT,    //INOUT: just to make sure that nobody use this buffer before the function is completed
             //  sizeof(double)*dwork_LD*nb,      dwork,    NODEP,
              sizeof(magma_int_t),             &dwork_LD,VALUE,
              /*dependency to check before sending A_src*/
              sizeof(void *),                  A_src_dep_ptr,INPUT,
              /*dependency to set after the transfer in dA_dst is completed*/
              sizeof(void *),                  dA_dst_dep_ptr,OUTPUT,
              0);

      //schedule_insert_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)
 }

  void magma_insert_dev_dsetmatrix_async_transpose(magma_int_t deviceID, magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD, magma_queue_t stream1, double *dwork, magma_int_t dwork_LD, void *A_src_dep_ptr, void *dA_dst_dep_ptr)
 {
     
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
   //  schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_setmatrix");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");

      schedule_Insert_Task(sched_obj, magma_task_dev_dsetmatrix_async_transpose, &task_flags,
          sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*LDA*nb,           A_src,    INPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(double)*dA_LD*nb,         dA_dst,   OUTPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              sizeof(magma_queue_t),           &stream1,   VALUE,
              sizeof(double)*dwork_LD*nb,      dwork,    INOUT,    //INOUT: just to make sure that nobody use this buffer before the function is completed
             //  sizeof(double)*dwork_LD*nb,      dwork,    NODEP,
              sizeof(magma_int_t),             &dwork_LD,VALUE,
              /*dependency to check before sending A_src*/
              sizeof(void *),                  A_src_dep_ptr,INPUT,
              /*dependency to set after the transfer in dA_dst is completed*/
              sizeof(void *),                  dA_dst_dep_ptr,OUTPUT,
              0);

      //schedule_insert_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)
 }


 void magma_insert_dev_dgetmatrix_transpose(magma_int_t deviceID, magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA, double *dwork, magma_int_t dwork_LD, void *A_dst_dep_ptr)               
 {
 Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

 schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
 //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
 //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
 schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

 schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_getmatrix");
 schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"indigo");

           schedule_Insert_Task(sched_obj, magma_task_dev_dgetmatrix_transpose, &task_flags,
               sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*dA_LD*nb,         dA_src,   INPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              sizeof(double)*LDA*nb,           A_dst,    OUTPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(double)*dwork_LD*nb,      dwork,    INOUT,    //just to make sure that nobody use this buffer before the function is completed
              //sizeof(double)*dwork_LD*nb,      dwork,   NODEP,    
              sizeof(magma_int_t),             &dwork_LD,VALUE,
              /*dependency to set after the transfer in dA_dst is completed*/
              sizeof(void *),                  A_dst_dep_ptr,INOUT,
              0);
     // schedule_insert_getpanel(gpu_nrows, ncols, dAT(K,K+A_N), dAT_LD, A(K,K+A_N), A_LD);
 }

  void magma_insert_dev_dgetmatrix_async_transpose(magma_int_t deviceID, magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA, magma_queue_t stream1, double *dwork, magma_int_t dwork_LD, void *A_dst_dep_ptr)               
 {
 Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

 schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
 //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
 //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
 schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

 schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_getmatrix");
 schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"indigo");

           schedule_Insert_Task(sched_obj, magma_task_dev_dgetmatrix_async_transpose, &task_flags,
               sizeof(magma_int_t),            &deviceID,VALUE,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*dA_LD*nb,         dA_src,   INPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              sizeof(double)*LDA*nb,           A_dst,    OUTPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(magma_queue_t),           &stream1,   VALUE,
              sizeof(double)*dwork_LD*nb,      dwork,    INOUT,    //just to make sure that nobody use this buffer before the function is completed
              //sizeof(double)*dwork_LD*nb,      dwork,   NODEP,    
              sizeof(magma_int_t),             &dwork_LD,VALUE,
              /*dependency to set after the transfer in dA_dst is completed*/
              sizeof(void *),                  A_dst_dep_ptr,INOUT,
              0);
     // schedule_insert_getpanel(gpu_nrows, ncols, dAT(K,K+A_N), dAT_LD, A(K,K+A_N), A_LD);
 }

void magma_insert_dev_dlaswp(magma_int_t deviceID, magma_int_t n, double *dA, magma_int_t lda, magma_int_t i1, magma_int_t i2, magma_int_t *ipiv, magma_int_t inci,  void *dA_dep_ptr)
{
            magma_int_t ipiv_size;
Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;
        ipiv_size = (inci>0)?i2*inci:-i2*inci;

        schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
    //    schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
    //    schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);

        schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

        schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dlaswp");
        schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"azure3");
        

        schedule_Insert_Task(sched_obj, magma_task_dev_dlaswp, &task_flags,
            sizeof(magma_int_t),             &deviceID,VALUE,
              sizeof(magma_int_t),             &n,       VALUE,
              sizeof(double)*lda*n,            dA,       INOUT,
              sizeof(magma_int_t),             &lda,     VALUE,
              sizeof(magma_int_t),             &i1,      VALUE,
              sizeof(magma_int_t),             &i2,      VALUE,
              sizeof(magma_int_t)*ipiv_size,   ipiv,     INPUT,
              sizeof(magma_int_t),             &inci,    VALUE,
              /*additional dependency to check before the swap*/
              sizeof(void *),                  dA_dep_ptr,INPUT,
              0);

     //magma_insert_dlaswp(gpu_ncols, dAT(K,K+A_N), dAT_LD, c_one, nb, &ipiv[K], c_one);
}
/**/
void magma_insert_dev_dtrsm(magma_int_t deviceID, magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n, double alpha, double *dA, magma_int_t lda, double *dB, magma_int_t ldb)
{
             magma_int_t k;
Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;
             k = (side=='L')?m:n;

             schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
      //       schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
     //        schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
             schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

             schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dtrsm");
             schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"pink");

             schedule_Insert_Task(sched_obj, magma_task_dev_dtrsm, &task_flags,
                 sizeof(magma_int_t),          &deviceID,VALUE,
              sizeof(magma_side_t),            &side,   VALUE,
              sizeof(magma_uplo_t),            &uplo,   VALUE,
              sizeof(magma_trans_t),           &trans,  VALUE,
              sizeof(magma_side_t),            &diag,   VALUE,
              sizeof(magma_int_t),             &m,      VALUE,
              sizeof(magma_int_t),             &n,      VALUE,
              sizeof(double),                  &alpha,  VALUE,
              sizeof(double)*lda*k,            dA,      INPUT,
              sizeof(magma_int_t),             &lda,    VALUE,
              sizeof(double)*ldb*n,            dB,      INOUT,
              sizeof(magma_int_t),             &ldb,    VALUE,
              0);
     //magma_insert_dtrsm('R', 'U', 'N', 'U', gpu_ncols, nb, c_one, dAT(K,K), dAT_LD, dAT(K,K+A_N), dAT_LD);
}

void magma_insert_dev_dgemm(magma_int_t deviceID, magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, double alpha, double *dA, magma_int_t lda, double *dB, magma_int_t ldb, double beta, double *dC, magma_int_t ldc)
{
     magma_int_t ka, kb;
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     ka = (transA==MagmaNoTrans)?k:m;
     kb = (transB==MagmaNoTrans)?n:k;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
   //  schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
   //  schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, (deviceID+1) ) ;

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dgemm");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"greenyellow");

              schedule_Insert_Task(sched_obj, magma_task_dev_dgemm, &task_flags,
                  sizeof(magma_int_t),         &deviceID,VALUE,
              sizeof(magma_trans_t),           &transA, VALUE,
              sizeof(magma_trans_t),           &transB, VALUE,
              sizeof(magma_int_t),             &m,      VALUE,
              sizeof(magma_int_t),             &n,      VALUE,
              sizeof(magma_int_t),             &k,      VALUE,
              sizeof(double),                  &alpha,  VALUE,
              sizeof(double)*lda*ka,           dA,      INPUT,
              sizeof(magma_int_t),             &lda,    VALUE,
              sizeof(double)*ldb*kb,           dB,      INPUT,
              sizeof(magma_int_t),             &ldb,    VALUE,
              sizeof(double),                  &beta,   VALUE,
              sizeof(double)*ldc*n,            dC,      INOUT,
              sizeof(magma_int_t),             &ldc,    VALUE,
              0);

     //magma_insert_dgemm('N','N', gpu_ncols, gpu_nrows, nb, c_neg_one, dAT(K,K+A_N), dAT_LD, dAT(K+1,K), dAT_LD, c_one, dAT(K+1,K+A_N), dAT_LD);
}

