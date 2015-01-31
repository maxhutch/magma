/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
#ifndef MAGMA_TASK_DEV_H
#define MAGMA_TASK_DEV_H

#include "schedule.h"

void magma_task_dev_set_compute_stream(int deviceID, magma_queue_t stream1);

void magma_task_dev_dmalloc_pinned(Schedule* sched_obj);
void magma_task_dev_dfree_pinned(Schedule* sched_obj);
void magma_task_dev_dfree_pinned_index(Schedule* sched_obj);

void magma_task_dev_queue_sync(Schedule* sched_obj);
void magma_task_dev_dsetmatrix(Schedule* sched_obj);
void magma_task_dev_dgetmatrix(Schedule* sched_obj);
void magma_task_dev_dsetmatrix_transpose(Schedule* sched_obj);
void magma_task_dev_dsetmatrix_async_transpose(Schedule* sched_obj);
void magma_task_dev_dgetmatrix_transpose(Schedule* sched_obj);  
void magma_task_dev_dgetmatrix_async_transpose(Schedule* sched_obj);               
void magma_task_dev_dlaswp(Schedule* sched_obj);
void magma_task_dev_dtrsm(Schedule* sched_obj);
void magma_task_dev_dgemm(Schedule* sched_obj);
void magma_task_dev_update(Schedule* sched_obj);
#endif
