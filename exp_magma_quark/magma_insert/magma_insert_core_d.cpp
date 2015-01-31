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

#include "magma_task_core_d.h"
#include "magma_task_d.h"

/*CPU wrapper*/
void magma_insert_core_dmemset(double *ptr, double value, magma_int_t n){

    Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

    schedule_Insert_Task(sched_obj, magma_task_core_dmemset, &task_flags,
        sizeof(double)*n,                ptr,            INOUT,
        sizeof(double),                  &value,         VALUE,
        sizeof(int),                     &n,             VALUE, 
        0);
}

void magma_insert_core_dgetrf(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, void *colptr)
{
     magma_int_t min_mn;
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     min_mn = min(m,n);

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_but_master_mask);

         schedule_Insert_Task(sched_obj, magma_task_core_dgetrf, &task_flags,
              sizeof(magma_int_t),             &m,       VALUE,
              sizeof(magma_int_t),             &n,       VALUE,
              sizeof(double)*LDA*n,            A,        INOUT,
              sizeof(magma_int_t),             &LDA,     VALUE,
              sizeof(magma_int_t)*min_mn,      ipiv,     OUTPUT,
              sizeof(magma_int_t),             info,     NODEP,
              /*dependency on the column that the panel write*/
              sizeof(void *),                  colptr,   INOUT,
              0);
     //magma_insert_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, &ipiv[0], iinfo);
}

void magma_insert_core_dgetrf_rec(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, magma_int_t num_threads, void *colptr)
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

    //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_panel_mask);
    schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_but_master_mask);

    
    schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dgetrf");
    schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"red");


    schedule_Insert_Task(sched_obj, magma_task_core_zgetrf_reclap, &task_flags,
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

void magma_insert_core_dtslu(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, magma_int_t num_threads, void *colptr)
{
    magma_int_t min_mn;
    
    min_mn = min(m,n);

    /*QUARK_CORE_zgetrf_reclap(...)*/

    Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

    
    //printf("m:%d, n:%d, Pr:%d\n",m,n,panel_thread_count);
    schedule_Task_Flag_Set(&task_flags, TASK_THREAD_COUNT, num_threads);
    schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);//schedule_TASK_MAX_PRIORITY - k 

    //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_panel_mask);
    schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_but_master_mask);

    
    schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dtslu");
    schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"red");


    schedule_Insert_Task(sched_obj, magma_task_core_dtslu, &task_flags,
        sizeof(int),                        &m,             VALUE,
        sizeof(int),                        &n,             VALUE,
        //sizeof(double)*nb*nb,   A,             INOUT,
        sizeof(double)*LDA*n,                A,              INOUT,
        sizeof(int),                        &LDA,           VALUE,
        sizeof(int)*min_mn,                 ipiv,           OUTPUT,
        sizeof(int),                        info,           VALUE,
        sizeof(int),                        &num_threads,   VALUE,
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

 void magma_insert_core_dlaswp(magma_int_t n, double *A, magma_int_t LDA, magma_int_t K1, magma_int_t K2, magma_int_t *ipiv, magma_int_t incx, void *colptr)
 {
        magma_int_t ipiv_size;
        Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

        ipiv_size = incx>0?K2*incx:-K2*incx;

        schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
        //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);
        schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_mask);

        schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dlaswp");
        schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"orange");

        schedule_Insert_Task(sched_obj, magma_task_core_dlaswp, &task_flags,
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

      //magma_insert_core_dlaswp(ncols, A(K,J), A_LD, c_one, nb, &ipiv[K], c_one);
 }
     
void  magma_insert_core_dtrsm(char side, char uplo, char transa, char diag, 
                            magma_int_t m, magma_int_t n, double alpha, 
                            double *A, magma_int_t LDA, double *B, magma_int_t LDB, void *colptr)
{
             magma_int_t k;
             Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

             k = (side=='L')?m:n;

             schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
             //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);
             schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_mask);

             schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dtrsm");
             schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"violet");

             schedule_Insert_Task(sched_obj, magma_task_core_dtrsm, &task_flags,
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
              sizeof(void *),                  colptr,  INOUT, //|GATHERV
              0);
    //magma_insert_core_dtrsm("L", "L", "N", "U", nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD);
}

/*The different with the classic insert_dtrsm is that it allows many dtrsm to be performed on the same column at once*/
void  magma_insert_core_dtrsm_gatherv(char side, char uplo, char transa, char diag, 
                            magma_int_t m, magma_int_t n, double alpha, 
                            double *A, magma_int_t LDA, double *B, magma_int_t LDB, void *colptr)
{
             magma_int_t k;
             Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

             k = (side=='L')?m:n;

             schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
             //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);
             schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_mask);

             schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dtrsm");
             schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"violet");

             schedule_Insert_Task(sched_obj, magma_task_core_dtrsm, &task_flags,
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
              sizeof(void *),                  colptr,  INOUT|GATHERV, //
              0);
    //magma_insert_core_dtrsm("L", "L", "N", "U", nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD);
}

void magma_insert_core_dgemm(char transa, char transb, 
                           magma_int_t m, magma_int_t n, magma_int_t k, double alpha, 
                           double *A, magma_int_t LDA, double *B, magma_int_t LDB, double beta, double *C, magma_int_t LDC, void *A_colptr, void *C_colptr)
 {
     magma_int_t ka, kb;
     Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

     ka = (transa==MagmaNoTrans)?k:m;
     kb = (transb=='N')?n:k;

     schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, current_priority);
     //schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,  (intptr_t) str_thread1_excluded_mask);
     schedule_Task_Flag_Set(&task_flags, TASK_LOCK_TO_THREAD_MASK,    (intptr_t) str_thread_computation_mask);

     schedule_Task_Flag_Set(&task_flags, TASK_LABEL, (intptr_t)"cpu_dgemm");
     schedule_Task_Flag_Set(&task_flags, TASK_COLOR, (intptr_t)"darkgreen");

              schedule_Insert_Task(sched_obj, magma_task_core_dgemm, &task_flags,
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
     //magma_insert_core_dgemm("N","N", nrows, ncols, nb, c_neg_one, A(I,K), A_LD, A(K,J), A_LD, c_one, A(I,J), A_LD);
 }
