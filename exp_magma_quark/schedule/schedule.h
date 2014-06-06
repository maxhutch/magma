/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
       @generated d Thu May 23 11:46:05 2013 
 
       Any scheduler can be used later using this wrapper.
*/

#ifndef SCHEDULE_H
#define SCHEDULE_H

/*if we link with quark then we wrap it here*/
#include "schedule_wrap_quark.h"

#include "schedule_insert_d.h"

/*Global var in schedule.cpp */

/*TODO: move in async_args for thread safety*/
extern Schedule *sched_obj;

extern int current_priority;

extern unsigned char *str_master_excluded_mask;
extern unsigned char *str_thread1_excluded_mask;
extern unsigned char *str_panel_mask;


/*scheduler wrapper*/
/* Initialize the scheduler*/
void schedule_init(int nbcores);

void schedule_barrier();

void schedule_delete();

void schedule_set_task_priority(int priority);

/*added functions*/
/*introduce a fake Read-write dependencies between 2 pointers*/
void schedule_insert_RW_dependencies(void *ptr1, void *ptr2);

/*introduce a fake Read-write dependencies between 2 pointers*/
void schedule_insert_gatherv_dependency(void *ptr1, void *ptr2);

#endif
