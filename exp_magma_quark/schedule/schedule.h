/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
/* 
       This file is a wrapper for any underlying scheduler.
*/
#ifndef SCHEDULE_H
#define SCHEDULE_H

/*if we link with quark then we wrap it here*/
#include "schedule_wrap_quark.h"

#include "magma_insert_d.h"

#include "magma_insert_dev_d.h"
/*Global var in schedule.cpp */

/*TODO: move in amc_args for thread safety*/
extern Schedule *sched_obj;

extern int current_priority;

//extern unsigned char *str_master_excluded_mask;
//extern unsigned char *str_thread1_excluded_mask;
//extern unsigned char *str_panel_mask;

extern unsigned char *str_thread_communication_mask;
extern unsigned char *str_thread_computation_mask;
extern unsigned char *str_thread_computation_but_master_mask;

/*scheduler wrapper*/
/* Initialize the scheduler*/
void magma_schedule_init(int nbcores, int ngpu);

void magma_schedule_barrier();

void magma_schedule_delete();

void magma_schedule_set_task_priority(int priority);

/*added functions*/
/*introduce a fake Read-write dependencies between 2 pointers*/
void magma_schedule_insert_RW_dependencies(void *ptr1, void *ptr2);

/*introduce a fake Read-write dependencies between 2 pointers*/
void magma_schedule_insert_gatherv_dependency(void *ptr1, void *ptr2);

#endif

