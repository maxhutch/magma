/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
#ifndef TASK_CORE_KERNEL
#define TASK_CORE_KERNEL

#include "schedule.h"

void task_core_void(Schedule* sched_obj);
/*Fill block of memory: sets the first n bytes of the block of memory pointed by ptr to the specified value*/
void task_core_dmemset(Schedule* sched_obj);
void task_core_dgetrf(Schedule* sched_obj);
//void task_core_dgetrf_rec(Schedule* sched_obj );
void task_core_zgetrf_reclap(Schedule* sched_obj);
void task_core_dlaswp(Schedule* sched_obj);   
void task_core_dtrsm(Schedule* sched_obj);
void task_core_dgemm(Schedule* sched_obj);

#endif
