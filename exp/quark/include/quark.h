/**
 *
 * @file quark.h
 *
 * Dynamic scheduler functions
 *
 * PLASMA is a software package provided by Univ. of Tennessee,
 * Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.3.1
 * @author Asim YarKhan
 * @date January 2015
 *
 **/

#ifndef QUARK_H
#define QUARK_H

#include <limits.h>
#include <stdio.h>

#if defined( _WIN32 )
  /* This must be included before INPUT is defined below, otherwise we
     have a name clash/problem  */
  #include <windows.h>
  #include <limits.h>
#else 
  #include <inttypes.h>
#endif

#include "quark_unpack_args.h"

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

//#define DBGPRINTF(args...) if (0) {};
/* #define DBGPRINTF(args...) { fprintf(stderr,"%s:%d: [%s] ",__FILE__,__LINE__,__FUNCTION__); fprintf(stderr, args); } */
//#define LOGPRINTF(args...) { fprintf(stderr,"%s:%d: [%s] ",__FILE__,__LINE__,__FUNCTION__); fprintf(stderr, args); }
// #define DBGPRINTF(args...) { fprintf(stderr, args); }

#define QUARK_SUCCESS 0
#define QUARK_ERR -1
#define QUARK_ERR_UNEXPECTED -1
#define QUARK_ERR_NOT_SUPPORTED -2

/* Data items can be: */
/*  INPUT, OUTPUT, INOUT:  these data items create dependencies */
/*  VALUE:  these data items get copied over */
/*  NODEP:  these data items get copied over, and are not used for dependencies */
/*  SCRATCH:  these data items can be allocated (and deallocted) by the scheduler when tasks execute  */
typedef enum { QINPUT=0x01, OUTPUT=0x02, INOUT=0x03, VALUE=0x04, NODEP=0x05, SCRATCH=0x06 } quark_direction_t;
#define INPUT 0x01

/* Data locality flag; ie keep data on the same core if possible */
#define LOCALITY ( 1 << 3 )
#define NOLOCALITY 0x00

/* A data address with a sequence of ACCUMULATOR dependencies will allow the related tasks to be reordered  */
#define ACCUMULATOR ( 1 << 4 )
#define NOACCUMULATOR 0x00

/* A data address with a sequence of GATHERV dependencies will allow the related tasks to be run in parallel  */
#define GATHERV ( 1 << 5 )
#define NOGATHERV 0x00

/* The following are task level flags, that can be either provided as additional arguments to the task, or via SET functions */
/* The task label; should be provided as a null terminated string */
#define TASK_LABEL ( 1 << 6 )
#define TASKLABEL ( 1 << 6 )    /* depreciated label */
/* The task color; should be provided as a null terminated string */
#define TASK_COLOR ( 1 << 7 )
#define TASKCOLOR ( 1 << 7 )    /* depreciated label */
/* The priority of the task, provided as an integer */
#define TASK_PRIORITY ( 1 << 8 )
/* Lock the task to a specific thread number (0 ... NTHREADS-1), provided as an integer */
#define TASK_LOCK_TO_THREAD ( 1 << 9 )
/* The sequence pointer to be associated with the task, provided as a pointer */
#define TASK_SEQUENCE ( 1 << 10 )
/* An integere for the number of threads require */
#define TASK_THREAD_COUNT ( 1 << 11 )
/* Lock the task to a thead mask (0 ... NTHREADS-1) bits long, provided as a character array (byte array) */
// #define TASK_LOCK_TO_THREAD_MASK ( 1 << 12 )

/* The range for priority values */
#define QUARK_TASK_MIN_PRIORITY 0
#define QUARK_TASK_MAX_PRIORITY INT_MAX

/* Definition of structure holding scheduler information */
typedef struct quark_s Quark;

/* Structure holding task information */
typedef struct quark_task_s Quark_Task;

/* Create a type for setting task flags */
struct quark_task_flags_s {
    int task_priority;
    int task_lock_to_thread;
    char *task_color;
    char *task_label;
    void *task_sequence;
    int task_thread_count;
//    char *task_lock_to_thread_mask;
};

typedef struct quark_task_flags_s Quark_Task_Flags;
/* Static initializer for Quark_Task_Flags_t */
#define Quark_Task_Flags_Initializer { (int)0, (int)-1, (char *)NULL, (char *)NULL, (void *)NULL, (int)1 }

/* Setup scheduler data structures, assumes threads are managed seperately */
Quark *QUARK_Setup(int num_threads);

/* Setup scheduler data structures, spawn worker threads, start the workers working  */
Quark *QUARK_New(int num_threads);

/* Add a task, called by the master process (thread_rank 0)  */
unsigned long long QUARK_Insert_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...);

/* Main work loop, called externally by everyone but the master
 * (master manages this internally to the insert_task and waitall
 * routines). Each worker thread can call work_main_loop( quark,
 * thread_rank), where thread rank is 1...NUMTHREADS ) */
void QUARK_Worker_Loop(Quark *quark, int thread_rank);

/* Finish work and return.  Workers do not exit */
void QUARK_Barrier(Quark * quark);

/* Just wait for current tasks to complete, the scheduler and
 * strutures remain as is... should allow for repeated use of the
 * scheduler.  The workers return from their loops.*/
void QUARK_Waitall(Quark * quark);

/* Delete scheduler, shutdown threads, finish everything, free structures */
void QUARK_Delete(Quark * quark);

/* Free scheduling data structures */
void QUARK_Free(Quark * quark);

/* Cancel a specific task */
int QUARK_Cancel_Task(Quark *quark, unsigned long long taskid);

/* Returns a pointer to the list of arguments, used when unpacking the
   arguments; Returna a pointer to icl_list_t, so icl_list.h will need
   bo included if you use this function */
void *QUARK_Args_List(Quark *quark);

/* Return a pointer to an argument.  The variable last_arg should be
   NULL on the first call, then each subsequent call will used
   last_arg to get the the next argument. */
void *QUARK_Args_Pop( void *args_list, void **last_arg);

/* Utility function returning rank of the current thread */
int QUARK_Thread_Rank(Quark *quark);

/* Packed task interface */
/* Create a task data structure to hold arguments */
Quark_Task *QUARK_Task_Init(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags );

/* Add (or pack) the arguments into a task data structure (make sure of the correct order) */
void QUARK_Task_Pack_Arg( Quark *quark, Quark_Task *task, int arg_size, void *arg_ptr, int arg_flags );

/* Insert the packed task data strucure into the scheduler for execution */
unsigned long long QUARK_Insert_Task_Packed(Quark * quark, Quark_Task *task );

/* Unsupported function for debugging purposes; execute task AT ONCE */
unsigned long long QUARK_Execute_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...);

/* Get the label (if any) associated with the current task; used for printing and debugging  */
char *QUARK_Get_Task_Label(Quark *quark);

/* Method for setting task flags */
Quark_Task_Flags *QUARK_Task_Flag_Set( Quark_Task_Flags *flags, int flag, intptr_t val );

/* Type for task sequences */
typedef struct Quark_sequence_s Quark_Sequence;

/* Create a seqeuence structure, to hold sequences of tasks */
Quark_Sequence *QUARK_Sequence_Create( Quark *quark );

/* Called by worker, cancel any pending tasks, and mark sequence so that it does not accept any more tasks */
int QUARK_Sequence_Cancel( Quark *quark, Quark_Sequence *sequence );

/* Destroy a sequence structure, cancelling any pending tasks */
Quark_Sequence *QUARK_Sequence_Destroy( Quark *quark, Quark_Sequence *sequence );

/* Wait for a sequence of tasks to complete */
int QUARK_Sequence_Wait( Quark *quark, Quark_Sequence *sequence );

/* Get the sequence information associated the current task/worker, this was provided when the tasks was created */
Quark_Sequence *QUARK_Get_Sequence(Quark *quark);


#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif                          /* QUARK.H */
