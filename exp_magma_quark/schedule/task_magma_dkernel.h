/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
#ifndef TASK_MAGMA_KERNEL
#define TASK_MAGMA_KERNEL

#include "schedule.h"

void task_magma_dmalloc_pinned(Schedule* sched_obj);
void task_magma_dfree_pinned(Schedule* sched_obj);
void task_magma_dfree_pinned_index(Schedule* sched_obj);
void task_magma_dsetmatrix(Schedule* sched_obj);
void task_magma_dgetmatrix(Schedule* sched_obj);
void task_magma_dsetmatrix_transpose(Schedule* sched_obj);
void task_magma_dgetmatrix_transpose(Schedule* sched_obj);               
void task_magma_dlaswp(Schedule* sched_obj);
void task_magma_dtrsm(Schedule* sched_obj);
void task_magma_dgemm(Schedule* sched_obj);
void task_magma_update(Schedule* sched_obj);
#endif
