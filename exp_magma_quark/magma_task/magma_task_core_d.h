/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
#ifndef MAGMA_TASK_CORE_D_H
#define MAGMA_TASK_CORE_D_H

#include "schedule.h"

void magma_task_core_void(Schedule* sched_obj);
/*Fill block of memory: sets the first n bytes of the block of memory pointed by ptr to the specified value*/
void magma_task_core_dmemset(Schedule* sched_obj);
void magma_task_core_dgetrf(Schedule* sched_obj);
//void magma_task_core_dgetrf_rec(Schedule* sched_obj );
void magma_task_core_zgetrf_reclap(Schedule* sched_obj);
void magma_task_core_dlaswp(Schedule* sched_obj);   
void magma_task_core_dtrsm(Schedule* sched_obj);
void magma_task_core_dgemm(Schedule* sched_obj);

void magma_task_core_dtslu(Schedule* sched_obj);
#endif

