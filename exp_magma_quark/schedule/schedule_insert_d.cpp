#include "common_magma.h"

#include "schedule.h"

#include "task_core_dkernel.h"
#include "task_magma_dkernel.h"

/*CPU wrapper*/
void schedule_insert_dmemset(double *ptr, double value, magma_int_t n){

    Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

    schedule_Insert_Task(sched_obj, task_core_dmemset, &task_flags,
        sizeof(double)*n,                ptr,            INOUT,
        sizeof(double),                  &value,         VALUE,
        sizeof(int),                     &n,             VALUE, 
        0);
}

void schedule_insert_dgetrf(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, void *colptr)
{
     magma_int_t min_mn;
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     min_mn = min(m,n);

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);

         schedule_Insert_Task(sched_obj, task_core_dgetrf, &task_flags,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &n,       VALUE,
              sizeof(double)*LDA*n,            A,        INOUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(magma_int_t)*min_mn,      ipiv,     OUTPUT,
              sizeof(magma_int_t),             info,     NODEP,
              /*dependency on the column that the panel write*/
              sizeof(void *),                  colptr,   INOUT,
              0);
     //schedule_insert_dgetrf_rec(A_m, nb, A(0,K), A_LD, &ipiv[0], iinfo);
}

void schedule_insert_dgetrf_rec(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, magma_int_t num_threads, void *colptr)
{
    magma_int_t min_mn;
    int panel_thread_count;
    min_mn = min(m,n);

    /*QUARK_CORE_zgetrf_reclap(...)*/

    Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

    //plasma = plasma_context_self();
    //PLASMA_Sequence_Create(&sequence);
    //QUARK_Task_Flag_Set(&task_flags, TASK_SEQUENCE, (intptr_t)sequence->quark_sequence);
    //QUARK_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);



    panel_thread_count = num_threads;
    
    while ( (panel_thread_count * 4 * n) > m ) {
            panel_thread_count--;
    }
    panel_thread_count = max(panel_thread_count,2);
    
    //printf("m:%d, n:%d, Pr:%d\n",m,n,panel_thread_count);
    schedule_Task_Flag_Set(&task_flags, TASK_THREAD_COUNT, panel_thread_count );
    schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);//schedule_TASK_MAX_PRIORITY - k 

    schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_panel_mask);

    schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dgetrf");
    schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"red");
    //plasma_dynamic_spawn();
    //CORE_zgetrf_reclap_init();

    /*
        magma_int_t min_mn;

        min_mn = min(m,n);

         schedule_Insert_Task(quark, task_core_dgetrf, 0,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &n,       VALUE,
              sizeof(double)*LDA*n,            A,        INOUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(magma_int_t)*min_mn,      ipiv,     OUTPUT,
              sizeof(magma_int_t),             info,     OUTPUT,
              0);
   */
    /*adapt recursif LU from plasma code*/
    /*
    schedule_CORE_zgetrf_reclap(quark, &task_flags,
                             m, n, n,
                             A, LDA, ipiv, 
                             *info,
                             num_threads);*/

    schedule_Insert_Task(sched_obj, task_core_zgetrf_reclap, &task_flags,
        sizeof(int),                        &m,             VALUE,
        sizeof(int),                        &n,             VALUE,
        //sizeof(double)*nb*nb,   A,             INOUT,
        sizeof(double)*LDA*n,                A,              INOUT,
        sizeof(int),                        &LDA,           VALUE,
        sizeof(int)*min_mn,                 ipiv,           OUTPUT,
        sizeof(int),                        info,           VALUE,
        sizeof(int),                        &panel_thread_count,   VALUE,
        /*dependency on the column that the panel write*/
        sizeof(void *),                     colptr,         INOUT,
        0);

}

/***************************************************************************//**
 *
 **/
/*
void schedule_CORE_zgetrf_reclap(Schedule *quark, Schedule_Task_Flags *task_flags,
                              int m, int n, int nb,
                              double *A, int lda,
                              int *IPIV,
                              int iinfo,
                              int nbthread)
{

    //DAG_CORE_GETRF;
    schedule_Insert_Task(quark, CORE_zgetrf_reclap_quark, task_flags,
        sizeof(int),                        &m,             VALUE,
        sizeof(int),                        &n,             VALUE,
        //sizeof(double)*nb*nb,   A,             INOUT,
        sizeof(double)*lda*n,   A,             INOUT,
        sizeof(int),                        &lda,           VALUE,
        sizeof(int)*nb,                     IPIV,          OUTPUT,
        sizeof(int),                        &iinfo,         VALUE,
        sizeof(int),                        &nbthread,      VALUE,
        0);
}
*/

 void schedule_insert_dlaswp(magma_int_t n, double *A, magma_int_t LDA, magma_int_t K1, magma_int_t K2, magma_int_t *ipiv, magma_int_t incx, void *colptr)
 {
        magma_int_t ipiv_size;
        Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

        ipiv_size = incx>0?K2*incx:-K2*incx;

        schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
        schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);

        schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dlaswp");
        schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"orange");

        schedule_Insert_Task(sched_obj, task_core_dlaswp, &task_flags,
              sizeof(magma_int_t),             &n,       VALUE,
              sizeof(double)*LDA*n,            A,        INOUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(magma_int_t),             &K1,      VALUE,
              sizeof(magma_int_t),             &K2,      VALUE,
              sizeof(magma_int_t)*ipiv_size,   ipiv,     INPUT,
              sizeof(magma_int_t),             &incx,    VALUE,
              /*additional dependency on the column that this swap read/modifies*/
              sizeof(void *),                  colptr,     INOUT,
              0);

      //schedule_insert_dlaswp(ncols, A(K,J), A_LD, c_one, nb, &ipiv[K], c_one);
 }
     
void  schedule_insert_dtrsm(char side, char uplo, char transa, char diag, 
                            magma_int_t m, magma_int_t n, double alpha, 
                            double *A, magma_int_t LDA, double *B, magma_int_t LDB, void *colptr)
{
             magma_int_t k;
             Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

             k = (side=='L')?m:n;

             schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
             schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);

             schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dtrsm");
             schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"violet");

             schedule_Insert_Task(sched_obj, task_core_dtrsm, &task_flags,
              sizeof(char),                    &side,   VALUE,
              sizeof(char),                    &uplo,   VALUE,
              sizeof(char),                    &transa, VALUE,
              sizeof(char),                    &diag,   VALUE,
              sizeof(magma_int_t),             &m,      VALUE,
              sizeof(magma_int_t),             &n,      VALUE,
              sizeof(double),                  &alpha,  VALUE,
              sizeof(double)*LDA*k,            A,       INPUT,
              sizeof(magma_int_t),             &LDA,    VALUE,
              sizeof(double)*LDB*n,            B,       INOUT,
              sizeof(magma_int_t),             &LDB,    VALUE,
              /*dependency on the column that this function read/modifies*/
              sizeof(void *),                  colptr,     INOUT,
              0);
    //schedule_insert_dtrsm("L", "L", "N", "U", nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD);
}

void schedule_insert_dgemm(char transa, char transb, 
                           magma_int_t m, magma_int_t n, magma_int_t k, double alpha, 
                           double *A, magma_int_t LDA, double *B, magma_int_t LDB, double beta, double *C, magma_int_t LDC, void *A_colptr, void *C_colptr)
 {
     magma_int_t ka, kb;
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     ka = (transa=='N')?k:m;
     kb = (transb=='N')?n:k;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dgemm");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"darkgreen");

              schedule_Insert_Task(sched_obj, task_core_dgemm, &task_flags,
              sizeof(char),                    &transa, VALUE,
              sizeof(char),                    &transb, VALUE,
              sizeof(magma_int_t),             &m,      VALUE,
              sizeof(magma_int_t),             &n,      VALUE,
              sizeof(magma_int_t),             &k,      VALUE,
              sizeof(double),                  &alpha,  VALUE,
              sizeof(double)*LDA*ka,           A,       INPUT,
              sizeof(magma_int_t),             &LDA,    VALUE,
              sizeof(double)*LDB*kb,           B,       INPUT,
              sizeof(magma_int_t),             &LDB,    VALUE,
              sizeof(double),                  &beta,   VALUE,
              sizeof(double)*LDC*n,            C,       INOUT,
              sizeof(magma_int_t),             &LDC,    VALUE,
              /*dependency on the column that this function read*/
              sizeof(void *),                  A_colptr,INPUT,
              /*dependency on the column that this function read/modifies*/
              sizeof(void *),                  C_colptr,INOUT |GATHERV,
              0);
     //schedule_insert_dgemm("N","N", nrows, ncols, nb, c_neg_one, A(I,K), A_LD, A(K,J), A_LD, c_one, A(I,J), A_LD);
 }
 
 /*CPU - GPU transfer wrapper*/
  void schedule_insert_magma_dmalloc_pinned(magma_int_t size, double **A, void *A_dep_ptr)
 {
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dmalloc_pinned");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");
     
      schedule_Insert_Task(sched_obj, task_magma_dmalloc_pinned, &task_flags,
              sizeof(magma_int_t),          &size,       VALUE,
              ///sizeof(double**),                A,            OUTPUT,//*size
              sizeof(double**),                A,            OUTPUT,//*size
              //dependency to set after the allocation is completed
              sizeof(void *),              A_dep_ptr,    INOUT,
              0);
      
      //magma_dmalloc_pinned(A, size);
      
 }

 void schedule_insert_magma_dfree_pinned(double *A, void *A_dep_ptr)
 {
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dfree_pinned");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");
     
      schedule_Insert_Task(sched_obj, task_magma_dfree_pinned, &task_flags,
              ///sizeof(double**),                A,            OUTPUT,//*size
              sizeof(double*),                    A,        INOUT,//*size
              
              //dependency to set after the allocation is completed
              sizeof(void *),              A_dep_ptr,    INOUT,
              0);
      
      //magma_dmalloc_pinned(A, size);
      
 }

 void schedule_insert_magma_dfree_pinned_index(double **A, int index, void *A_dep_ptr)
 {
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     //schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dfree_pinned_index");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");
     
      schedule_Insert_Task(sched_obj, task_magma_dfree_pinned_index, &task_flags,
              ///sizeof(double**),                A,            OUTPUT,//*size
              sizeof(double**),                A,              NODEP,//*size
              sizeof(magma_int_t),          &index,       VALUE,
              //dependency to set after the allocation is completed
              sizeof(void *),              A_dep_ptr,    INOUT,
              0);
      
      //magma_dmalloc_pinned(A, size);
      
 }

  void schedule_insert_magma_dsetmatrix(magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD)
 {

      schedule_Insert_Task(sched_obj, task_magma_dsetmatrix, 0,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*LDA*nb,           A_src,    INPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(double)*dA_LD*nb,         dA_dst,   OUTPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              0);

      //schedule_insert_setpanel(A_m, nb, A(0,K), A_LD, dA(0,K), dA_LD)
 }

 void schedule_insert_magma_dgetmatrix(magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA)               
 {
 
           schedule_Insert_Task(sched_obj, task_magma_dgetmatrix, 0,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &nb,      VALUE,
              sizeof(double)*dA_LD*nb,         dA_src,   INPUT,
              sizeof(magma_int_t),             &dA_LD,   VALUE,
              sizeof(double)*LDA*nb,           A_dst,    OUTPUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              0);
     // schedule_insert_getpanel(gpu_nrows, ncols, dAT(K,K+A_N), dAT_LD, A(K,K+A_N), A_LD);
 }

 void schedule_insert_magma_dsetmatrix_transpose(magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD, double *dwork, magma_int_t dwork_LD, void *A_src_dep_ptr, void *dA_dst_dep_ptr)
 {
     
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
    //  schedule_Task_Flag_Set (&task_flags, THREAD_SET_TO MANUAL_SCHEDULING, 1 ) ;
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_setmatrix");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"blue");

      schedule_Insert_Task(sched_obj, task_magma_dsetmatrix_transpose, &task_flags,
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

 void schedule_insert_magma_dgetmatrix_transpose(magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA, double *dwork, magma_int_t dwork_LD, void *A_dst_dep_ptr)               
 {
 Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

 schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
 schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
 schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
 schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_getmatrix");
 schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"indigo");

           schedule_Insert_Task(sched_obj, task_magma_dgetmatrix_transpose, &task_flags,
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

void schedule_insert_magma_dlaswp( magma_int_t n, double *dA, magma_int_t lda, magma_int_t i1, magma_int_t i2, magma_int_t *ipiv, magma_int_t inci,  void *dA_dep_ptr)
{
            magma_int_t ipiv_size;
Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;
        ipiv_size = (inci>0)?i2*inci:-i2*inci;

        schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
        schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
        schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);

        schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dlaswp");
        schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"azure3");
        

        schedule_Insert_Task(sched_obj, task_magma_dlaswp, &task_flags,
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

     //schedule_insert_magma_dlaswp(gpu_ncols, dAT(K,K+A_N), dAT_LD, c_one, nb, &ipiv[K], c_one);
}
/**/
void schedule_insert_magma_dtrsm(char side, char uplo, char trans, char diag, magma_int_t m, magma_int_t n, double alpha, double *dA, magma_int_t lda, double *dB, magma_int_t ldb )
{
             magma_int_t k;
Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;
             k = (side=='L')?m:n;

             schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
             schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);
             schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;

             schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dtrsm");
             schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"pink");

             schedule_Insert_Task(sched_obj, task_magma_dtrsm, &task_flags,
              sizeof(char),                    &side,   VALUE,
              sizeof(char),                    &uplo,   VALUE,
              sizeof(char),                    &trans,  VALUE,
              sizeof(char),                    &diag,   VALUE,
              sizeof(magma_int_t),             &m,      VALUE,
              sizeof(magma_int_t),             &n,      VALUE,
              sizeof(double),                  &alpha,  VALUE,
              sizeof(double)*lda*k,            dA,      INPUT,
              sizeof(magma_int_t),             &lda,    VALUE,
              sizeof(double)*ldb*n,            dB,      INOUT,
              sizeof(magma_int_t),             &ldb,    VALUE,
              0);
     //schedule_insert_magma_dtrsm('R', 'U', 'N', 'U', gpu_ncols, nb, c_one, dAT(K,K), dAT_LD, dAT(K,K+A_N), dAT_LD);
}

void schedule_insert_magma_dgemm(char transA, char transB, magma_int_t m, magma_int_t n, magma_int_t k, double alpha, double *dA, magma_int_t lda, double *dB, magma_int_t ldb, double beta, double *dC, magma_int_t ldc )
{
     magma_int_t ka, kb;
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     ka = (transA=='N')?k:m;
     kb = (transB=='N')?n:k;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set (&task_flags, TASK_LOCK_TO_THREAD, 1 ) ;
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_master_excluded_mask);

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"gpu_dgemm");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"greenyellow");

              schedule_Insert_Task(sched_obj, task_magma_dgemm, &task_flags,
              sizeof(char),                    &transA, VALUE,
              sizeof(char),                    &transB, VALUE,
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

     //schedule_insert_magma_dgemm('N','N', gpu_ncols, gpu_nrows, nb, c_neg_one, dAT(K,K+A_N), dAT_LD, dAT(K+1,K), dAT_LD, c_one, dAT(K+1,K+A_N), dAT_LD);
}
