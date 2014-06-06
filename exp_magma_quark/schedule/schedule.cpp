/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
       
       * This file is supposed to be a wrapper for an underlying scheduler
       * Actually, it uses quark as scheduler by default, but any scheduler can be used later by surcharging ALL of these functions
       *
       @author: Simplice Donfack 
 
*/

#include <math.h>
#include <stdlib.h>


#include "common_magma.h"


#include "schedule.h"

#include "task_core_dkernel.h"

/*Global var*/

/*TODO: move in async_args for thread safety*/
/*Scheduling objet*/
Schedule *sched_obj;

int current_priority;

unsigned char *str_master_excluded_mask;
unsigned char *str_thread1_excluded_mask;
unsigned char *str_panel_mask;

//extern void CORE_zgetrf_reclap_init(void);
/*void QUARK_CORE_zgetrf_reclap(Schedule *quark, Schedule_Task_Flags *task_flags,
                              int m, int n, int nb,
                              double *A, int lda,
                              int *IPIV,
                              int iinfo,
                              int nbthread);
*/
//void CORE_zgetrf_reclap_quark(Schedule* quark);




/* Initialize the scheduler*/
void schedule_init(int nbcores)
{
    int i,nbites;
     /*Initialize Schedule*/
     sched_obj = schedule_New(nbcores);

     current_priority = SCHEDULE_TASK_MAX_PRIORITY;
     nbites = (int) (nbcores/8)+1;
     str_master_excluded_mask = (unsigned char*) malloc(nbites*sizeof(unsigned char));
     str_thread1_excluded_mask = (unsigned char*) malloc(nbites*sizeof(unsigned char));
     str_panel_mask = (unsigned char*) malloc(nbites*sizeof(unsigned char));
     for(i=0;i<=nbcores-1;i++)
     {
         schedule_Bit_Set(str_master_excluded_mask, i, 1);
         schedule_Bit_Set(str_thread1_excluded_mask, i, 1);
         schedule_Bit_Set(str_panel_mask, i, 1);

         //printf("%d",schedule_Bit_Get(unsigned char *set, int number) str_master_excluded_mask[i])
     }

     //exclude master
     schedule_Bit_Set(str_master_excluded_mask, 0, 0);

     //exclude thread1 for panel and dgemm
     schedule_Bit_Set(str_thread1_excluded_mask, 1, 0);
     schedule_Bit_Set(str_panel_mask, 1, 0);

     /*exclude Panel threads from communicating*/
     /*
     for(i=1;i<=nbcores-1;i++) //Pr; min(Pr,nbcores-1)
     {

            schedule_Bit_Set(str_panel_mask, i, 1);
     }
     */
     //for(i=min(Pr,nbcores-1)+4;i<=nbcores-1;i++) //min(Pr,nbcores-1)
     //{
        //    schedule_Bit_Set(str_master_excluded_mask, i, 1);

        //    //schedule_Bit_Set(str_panel_mask, i, 1);
     //}

     //exclude first 4 threads
     //printf("S:%d\n",str_master_excluded_mask);
}

void schedule_barrier()
{
     schedule_Barrier(sched_obj);
}

void schedule_delete()
{
     schedule_Delete(sched_obj);
     free(str_master_excluded_mask);
     free(str_panel_mask);
     free(str_thread1_excluded_mask);
}

void schedule_set_task_priority(int priority)
{
    current_priority = priority;
}

/*added functions*/
/*introduce a fake Read-write dependencies between 2 pointers*/
void schedule_insert_RW_dependencies(void *ptr1, void *ptr2)
{

             schedule_Insert_Task(sched_obj, task_core_void, 0,
              sizeof(void*),             ptr1,     INPUT,
              sizeof(void*),             ptr2,     OUTPUT,
              0);
}

/*introduce a fake Read-write dependencies between 2 pointers, *ptr1 is collected into *ptr2*/
void schedule_insert_gatherv_dependency(void *ptr1, void *ptr2)
{
            Schedule_Task_Flags  task_flags = Schedule_Task_Flags_Initializer;

            schedule_Task_Flag_Set(&task_flags, TASK_PRIORITY, SCHEDULE_TASK_MAX_PRIORITY);//SCHEDULE_TASK_MAX_PRIORITY - k 

             schedule_Insert_Task(sched_obj, task_core_void, &task_flags,
              sizeof(void*),             ptr1,     INPUT,
              sizeof(void*),             ptr2,     OUTPUT|GATHERV,  ///|GATHERV
              0);
}




