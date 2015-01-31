/* **************************************************************************** */
/**
 * @file quark.c
 *
 * QUARK (QUeuing And Runtime for Kernels) provides a runtime
 * enviornment for the dynamic execution of precedence-constrained
 * tasks.
 *
 * QUARK is a software package provided by Univ. of Tennessee,
 * Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 2.3.1
 * @author Asim YarKhan
 * @date January 2015
 *
 */

/**
 * @defgroup QUARK QUARK: QUeuing And Runtime for Kernels
 *
 * These functions are available from the QUARK library for the
 * scheduling of kernel routines.
 */

/**
 * @defgroup QUARK_Unsupported QUARK: Unsupported functions
 *
 * These functions are used by internal QUARK and PLASMA developers to
 * obtain very specific behavior, but are unsupported and may have
 * unexpected results.
 */

/* ****************************************************************************

Summary of environment flags:

Change the window size (default should be checked in the code)
export QUARK_UNROLL_TASKS_PER_THREAD=num

Enable WAR avoidance (false dependency handling) (default=0 off)
export QUARK_WAR_DEPENDENCIES_ENABLE=1

Enable DAG generation (default=0 off)
export QUARK_DOT_DAG_ENABLE=1

**************************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

#ifndef inline
#define inline __inline
#endif

#if defined( _WIN32 ) || defined( _WIN64 )
#  define fopen(ppfile, name, mode) fopen_s(ppfile, name, mode)
#  define strdup _strdup
#  include "quarkwinthread.h"
#else
#  define fopen(ppfile, name, mode) *ppfile = fopen(name, mode)
#  include <pthread.h>
#endif

#ifdef TRUE
#undef TRUE
#endif

#ifdef FALSE
#undef FALSE
#endif

#include "icl_list.h"
#include "icl_hash.h"
#include "bsd_queue.h"
#include "bsd_tree.h"
#include "quark.h"
#include "quark_unpack_args.h"

#ifndef ULLONG_MAX
# define ULLONG_MAX 18446744073709551615ULL
#endif

/* External functions */
int quark_getenv_int(char* name, int defval);

#define DIRECTION_MASK 0x07
typedef enum { NOTREADY, QUEUED, RUNNING, DONE, CANCELLED } task_status;
typedef enum { DGETRF, DTSTRF, DGESSM, DSSSM } task_num;
typedef enum { FALSE, TRUE } bool;

struct quark_s {
    int low_water_mark;
    int high_water_mark;
    int num_threads;              /* number of threads */
    struct worker_s **worker;     /* array of workers [num_threads] */
    int *coresbind;               /* array of indices where to bind workers [num_threads] */
    volatile int list_robin;      /* round-robin list insertion index */
    volatile bool start;          /* start flag */
    volatile bool all_tasks_queued; /* flag */
    volatile long long num_tasks; /* number of tasks queued */
    icl_hash_t *task_set;
    pthread_mutex_t task_set_mutex;
    icl_hash_t *address_set;      /* hash table of addresses */
    pthread_mutex_t address_set_mutex;    /* hash table access mutex */
    pthread_attr_t thread_attr;   /* threads' attributes */
    int (*rank)();
    volatile int num_queued_tasks;
    pthread_cond_t num_queued_tasks_cond;
    int war_dependencies_enable;
#define tasklevel_width_max_level 5000
    int dot_dag_enable;
    int queue_before_computing;
    int tasklevel_width[tasklevel_width_max_level];
    pthread_mutex_t dot_dag_mutex;
    pthread_mutex_t completed_tasks_mutex;
    struct completed_tasks_head_s *completed_tasks;
};

struct Quark_sequence_s {
    volatile int status;
    pthread_mutex_t sequence_mutex;
    struct ll_list_head_s *tasks_in_sequence;
};

typedef struct worker_s {
    pthread_t thread_id;
    pthread_mutex_t ready_list_mutex;
    struct task_priority_tree_head_s *ready_list;
    volatile int ready_list_size;
    Quark_Task *current_task_ptr;
    Quark *quark_ptr;
    volatile bool finalize;       /* termination flag */
    volatile bool executing_task;
} Worker;

typedef struct quark_task_s {
    pthread_mutex_t task_mutex;
    void (*function) (Quark *);    /* task function pointer */
    volatile task_status status; /* Status of task; NOTREADY, READY; QUEUED; DONE */
    volatile int num_dependencies; /* number of dependencies */
    volatile int num_dependencies_remaining; /* number of dependencies remaining to be fulfilled */
    icl_list_t *args_list;        /* list of arguments (copies of scalar values and pointers) */
    icl_list_t *dependency_list;  /* list of dependencies */
    icl_list_t *scratch_list;        /* List of scratch space information and their sizes */
    volatile struct dependency_s *locality_preserving_dep; /* Try to run task on core that preserves the locality of this dependency */
    unsigned long long taskid; /* An identifier, used only for generating DAGs */
    unsigned long long tasklevel; /* An identifier, used only for generating DAGs */
    int lock_to_thread;
    char *task_label;            /* Label for this task, used in dot_dag generation */
    char *task_color;            /* Color for this task, used in dot_dag generation */
    int priority;                    /* Is this a high priority task */
    Quark_Sequence *sequence;
    struct ll_list_node_s *ptr_to_task_in_sequence; /* convenience pointer to this task in the sequence */
    int task_thread_count;                /* Num of threads required by task */
} Task;

typedef struct dependency_s {
    struct quark_task_s *task; /* pointer to parent task containing this dependency */
    void *address;              /* address of data */
    int size;                   /* Size of dependency data */
    quark_direction_t direction; /* direction of this dependency, INPUT, INOUT, OUTPUT */
    bool locality; /* Priority of this dependency; more like data locality */
    volatile bool accumulator; /* Tasks depending on this may be reordered, they accumulate results */
    bool gatherv; /* Tasks depending on this may be run in parallel, assured by the programmer */
    struct address_set_node_s *address_set_node_ptr; /* convenience pointer to address_set_node */
    icl_list_t *address_set_waiting_deps_node_ptr; /* convenience pointer to address_set_node waiting_deps node */
    icl_list_t *task_args_list_node_ptr; /* convenience ptr to the task->args_list [node] to use for WAR address updates */
    icl_list_t *task_dependency_list_node_ptr; /* convenience ptr to the task->dependency_list [node] */
    volatile bool ready;        /* Data dependency is ready */
} Dependency;

typedef struct scratch_s {
    void *ptr;                  /* address of scratch space */
    int size;                   /* Size of scratch data */
    icl_list_t *task_args_list_node_ptr; /* convenience ptr to the task->args_list [node] */
} Scratch;

typedef struct address_set_node_s {
    void *address; /* copy of key to the address_set - pointer to the data */
    int size;            /* data object size */
    volatile int last_thread; /* last thread to use this data - for scheduling/locality */
    icl_list_t *waiting_deps;    /* list of dependencies waiting for this data */
    volatile int num_waiting_input;    /* count of input dependencies for this data */
    volatile int num_waiting_output;    /* count of output dependencies for this data */
    volatile int num_waiting_inout;    /* count of inout dependencies for this data */
    volatile bool delete_data_at_address_when_node_is_deleted; /* used when data is copied in order to handle false dependencies  */
    unsigned long long last_writer_taskid; /* used for generating DOT DAGs */
    unsigned long long last_writer_tasklevel; /* used for tracking critical depth */
    unsigned long long last_reader_or_writer_taskid; /* used for generating DOT DAGs */
    unsigned long long last_reader_or_writer_tasklevel; /* used for tracking critical depth */
} Address_Set_Node;

/* Data structure for a list containing long long int values.  Used to
 * track task ids in sequences of tasks, so that the tasks in a
 * sequence can be controlled */
typedef struct ll_list_node_s {
    long long int val;
    LIST_ENTRY( ll_list_node_s ) entries;
} ll_list_node_t;
LIST_HEAD(ll_list_head_s, ll_list_node_s);
typedef struct ll_list_head_s ll_list_head_t;

typedef struct completed_tasks_node_s {
    Task *task;
    int workerid;
    TAILQ_ENTRY( completed_tasks_node_s ) entries;
} completed_tasks_node_t;
TAILQ_HEAD( completed_tasks_head_s, completed_tasks_node_s );
typedef struct completed_tasks_head_s completed_tasks_head_t;

/* Tree (red-black) structure for keeping a priority list of
 * executable tasks */
typedef struct task_priority_tree_node_s {
    int priority;
    Task *task;
    RB_ENTRY( task_priority_tree_node_s ) n_entry;
} task_priority_tree_node_t;
RB_HEAD( task_priority_tree_head_s, task_priority_tree_node_s );
typedef struct task_priority_tree_head_s task_priority_tree_head_t;
static int compare_task_priority_tree_nodes( task_priority_tree_node_t *n1, task_priority_tree_node_t *n2 )
{
    return n2->priority - n1->priority;
}
/* Generate red-black tree functions */
RB_GENERATE( task_priority_tree_head_s, task_priority_tree_node_s, n_entry, compare_task_priority_tree_nodes );


/* **************************************************************************** */
/**
 * Local function prototypes, declared static so they are not
 * available outside the scope of this file.
 */
static Task *quark_task_new();
static void task_delete( Quark *quark, Task *task);
static Worker *worker_new(Quark *quark, int rank);
static void worker_delete(Worker *worker);
static inline int quark_revolve_robin(Quark * quark);
static void quark_insert_task_dependencies(Quark * quark, Task * task);
static void quark_check_and_queue_ready_task( Quark *quark, Task *task );
static void work_set_affinity_and_call_main_loop(Worker *worker);
static void work_main_loop(Worker *worker);
static Scratch *scratch_new( void *arg_ptr, int arg_size, icl_list_t *task_args_list_node_ptr);
static void scratch_allocate( Task *task );
static void scratch_deallocate( Task *task );
static void address_set_node_delete( Quark *quark, Address_Set_Node *address_set_node );

/* static void worker_remove_completed_task_and_check_for_ready(Quark *quark, Task *task, int exe_thread_idx); */
static void worker_remove_completed_task_enqueue_for_later_processing(Quark *quark, Task *task, int worker_rank);
static void remove_completed_task_and_check_for_ready(Quark *quark, Task *task, int worker_rank);
static void process_completed_tasks(Quark *quark);

int  quark_setaffinity(int rank);
void quark_topology_init();
void quark_topology_finalize();
int  quark_get_numthreads();
int  *quark_get_affthreads();
int  quark_yield();

/* **************************************************************************** */
/**
 * Mutex wrappers for tracing/timing purposes.  Makes it easier to
 * profile the costs of these pthreads routines.
 */
inline static int pthread_mutex_lock_asn(pthread_mutex_t *mtx) { return pthread_mutex_lock( mtx ); }
inline static int pthread_mutex_trylock_asn(pthread_mutex_t *mtx) { return pthread_mutex_trylock( mtx ); }
inline static int pthread_mutex_unlock_asn(pthread_mutex_t *mtx) { return pthread_mutex_unlock( mtx ); }

inline static int pthread_mutex_lock_ready_list(pthread_mutex_t *mtx) { return pthread_mutex_lock( mtx ); }
inline static int pthread_mutex_trylock_ready_list(pthread_mutex_t *mtx) { return pthread_mutex_trylock( mtx ); }
inline static int pthread_mutex_unlock_ready_list(pthread_mutex_t *mtx) { return pthread_mutex_unlock( mtx ); }

inline static int pthread_mutex_lock_wrap(pthread_mutex_t *mtx) { return pthread_mutex_lock( mtx ); }
inline static int pthread_mutex_unlock_wrap(pthread_mutex_t *mtx) { return pthread_mutex_unlock( mtx ); }

inline static int pthread_mutex_lock_completed_tasks(pthread_mutex_t *mtx) { return pthread_mutex_lock( mtx ); }
inline static int pthread_mutex_trylock_completed_tasks(pthread_mutex_t *mtx) { return pthread_mutex_trylock( mtx ); }
inline static int pthread_mutex_unlock_completed_tasks(pthread_mutex_t *mtx) { return pthread_mutex_unlock( mtx ); }

inline static int pthread_cond_wait_ready_list( pthread_cond_t *cond, pthread_mutex_t *mtx ) { return pthread_cond_wait( cond, mtx ); }

/* **************************************************************************** */

/* If dags are to be generated, setup file name and pointer and
 * various macros.  This assumes that the fprintf function is thread
 * safe.  */
static char *quark_task_default_label = " ";
static char *quark_task_default_color = "white";
#define DEPCOLOR "black"
#define ANTIDEPCOLOR "red"
#define GATHERVDEPCOLOR "green"
#define DOT_DAG_FILENAME "dot_dag_file.dot"
FILE *dot_dag_file;
#define dot_dag_level_update( parent_level, child_level, quark )       \
    if ( quark->dot_dag_enable ) {                                     \
        pthread_mutex_lock_wrap( &quark->dot_dag_mutex );                   \
        child_level = (parent_level+1 < child_level ? child_level : parent_level+1 );  \
        pthread_mutex_unlock_wrap( &quark->dot_dag_mutex ); }
#define dot_dag_print_edge( parentid, childid, color)                 \
    if ( quark->dot_dag_enable && parentid!=0 ) {                      \
        pthread_mutex_lock_wrap( &quark->dot_dag_mutex );                   \
        fprintf(dot_dag_file, "t%lld->t%lld [color=\"%s\"];\n", parentid, childid, color); \
        pthread_mutex_unlock_wrap( &quark->dot_dag_mutex );                 \
    }

/* **************************************************************************** */
/**
 * Initialize the task data structure
 */
static Task *quark_task_new()
{
    static unsigned long long taskid = 1;
    Task *task = (Task *)malloc(sizeof(Task));
    assert(task != NULL);
    task->function = NULL;
    task->num_dependencies = 0;
    task->num_dependencies_remaining = 0;
    task->args_list = icl_list_new();
    assert(task->args_list != NULL);
    task->dependency_list = icl_list_new();
    assert(task->dependency_list != NULL);
    task->locality_preserving_dep = NULL;
    task->status = NOTREADY;
    task->scratch_list = icl_list_new();
    assert( task->scratch_list != NULL);
    assert( taskid < ULLONG_MAX );
    task->taskid = taskid++;
    task->tasklevel = 0;
    pthread_mutex_init(&task->task_mutex, NULL);
    task->ptr_to_task_in_sequence = NULL;
    task->sequence = NULL;
    task->priority = QUARK_TASK_MIN_PRIORITY;
    task->task_label = quark_task_default_label;
    task->task_color = quark_task_default_color;
    task->lock_to_thread = -1;
    task->task_thread_count = 1;
    return task;
}

/* **************************************************************************** */
/**
 * Free the task data structure
 */
static void task_delete(Quark *quark, Task *task)
{
    pthread_mutex_lock_wrap( &quark->task_set_mutex );
    icl_hash_delete( quark->task_set, &task->taskid, NULL, NULL );
    pthread_mutex_lock_wrap( &task->task_mutex );
    pthread_mutex_unlock_wrap( &quark->task_set_mutex );
    icl_list_destroy(task->args_list, free);
    icl_list_destroy(task->dependency_list, free);
    icl_list_destroy(task->scratch_list, free);
    if ( task->ptr_to_task_in_sequence != NULL ) {
        pthread_mutex_lock_wrap( &task->sequence->sequence_mutex );
        LIST_REMOVE( task->ptr_to_task_in_sequence, entries );
        pthread_mutex_unlock_wrap( &task->sequence->sequence_mutex );
        free( task->ptr_to_task_in_sequence );
    }
    if (task->task_color!=NULL && task->task_color!=quark_task_default_color) free(task->task_color);
    if (task->task_label!=NULL && task->task_label!=quark_task_default_label) free(task->task_label);
    pthread_mutex_unlock_wrap( &task->task_mutex );
    pthread_mutex_destroy( &task->task_mutex );
    free( task );
    // TODO pthread_mutex_lock_asn( &quark->address_set_mutex );
    quark->num_tasks--;
    // TODO pthread_mutex_unlock_asn( &quark->address_set_mutex );
}

/* **************************************************************************** */
/**
 * Return the rank of a thread
 */
int QUARK_Thread_Rank(Quark *quark)
{
    pthread_t self_id = pthread_self();
    int  i;
    for (i=0; i<quark->num_threads; i++)
        if (pthread_equal(quark->worker[i]->thread_id, self_id))
            return i;
    return -1;
}

/* **************************************************************************** */
/**
 * Return a pointer to the argument list being processed by the
 * current task and worker.
 *
 * @param[in] quark
 *         The scheduler's main data structure.
 * @return
 *          Pointer to the current argument list (icl_list_t *)
 * @ingroup QUARK
 */
void *QUARK_Args_List(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    assert( curr_task != NULL );
    return (void *)curr_task->args_list;
}

/* **************************************************************************** */
/**
 * Return a pointer to the next argument.  The variable last_arg
 * should be NULL on the first call, then each subsequent call will
 * used last_arg to get the the next argument. The argument list is
 * not actually popped, it is preservered intact.
 *
 * @param[in] args_list
 *         Pointer to the current arguments
 * @param[inout] last_arg
 *         Pointer to the last argument; should be NULL on the first call
 * @return
 *          Pointer to the next argument
 * @ingroup QUARK
 */
void *QUARK_Args_Pop( void *args_list, void **last_arg)
{
    icl_list_t *args = (icl_list_t *)args_list;
    icl_list_t *node = (icl_list_t *)*last_arg;
    void *arg = NULL;
    if ( node == NULL ) {
        node = icl_list_first( args );
        if (node!=NULL) arg = node->data;
    } else {
        node = icl_list_next( args, node );
        if (node!=NULL) arg = node->data;
    }
    *last_arg = node;
    return arg;
}

/* **************************************************************************** */
/**
 * Well known hash function: Fowler/Noll/Vo - 32 bit version
 */
static inline unsigned int fnv_hash_function( void *key, int len )
{
    unsigned char *p = key;
    unsigned int h = 2166136261u;
    int i;
    for ( i = 0; i < len; i++ )
        h = ( h * 16777619 ) ^ p[i];
    return h;
}

/* **************************************************************************** */
/**
 * Hash function to map addresses, cut into "long" size chunks, then
 * XOR. The result will be matched to hash table size using mod in the
 * hash table implementation
 */
static inline unsigned int address_hash_function(void *address)
{
    int len = sizeof(void *);
    unsigned int hashval = fnv_hash_function( &address, len );
    return hashval;
}

/* **************************************************************************** */
/**
 * Adress compare function for hash table */
static inline int address_key_compare(void *addr1, void *addr2)
{
    return (addr1 == addr2);
}

/* **************************************************************************** */
/**
 * Hash function for unsigned long longs (used for taskid)
 */
static inline unsigned int ullong_hash_function( void *key )
{
    int len = sizeof(unsigned long long);
    unsigned int hashval = fnv_hash_function( key, len );
    return hashval;
}
/* **************************************************************************** */
/**
 * Compare unsigned long longs for hash keys (used for taskid)
 */
static inline int ullong_key_compare( void *key1, void *key2  )
{
    return ( *(unsigned long long*)key1 == *(unsigned long long*)key2 );
}

/* **************************************************************************** */
/**
 * Rotate the next worker queue that will get a task assigned to it.
 * The master (0) never gets round-robin tasks assigned to it.
 */
static inline int quark_revolve_robin(Quark * quark)
{
    quark->list_robin++;
    if (quark->list_robin == quark->num_threads)
        quark->list_robin = 0;
    if (quark->list_robin==0 && quark->num_threads>1)
        quark->list_robin = 1;
    return quark->list_robin;
}

/* **************************************************************************** */
/**
 * Duplicate the argument, allocating a memory buffer for it
 */
static inline char *arg_dup(char *arg, int size)
{
    char *argbuf = (char *) malloc(size);
    assert( argbuf != NULL );
    memcpy(argbuf, arg, size);
    return argbuf;
}

/* **************************************************************************** */
/**
 * Allocate and initialize a dependency structure
 */
static inline Dependency *dependency_new(void *addr, long long size, quark_direction_t dir, bool loc, Task *task, bool accumulator, bool gatherv, icl_list_t *task_args_list_node_ptr)
{
    Dependency *dep = (Dependency *) malloc(sizeof(Dependency));
    assert(dep != NULL);
    dep->task = task;
    dep->address = addr;
    dep->size = size;
    dep->direction = dir;
    dep->locality = loc;
    dep->accumulator = accumulator;
    dep->gatherv = gatherv;
    dep->address_set_node_ptr = NULL; /* convenience ptr, filled later */
    dep->address_set_waiting_deps_node_ptr = NULL; /* convenience ptr, filled later */
    dep->task_args_list_node_ptr = task_args_list_node_ptr; /* convenience ptr for WAR address updating */
    dep->task_dependency_list_node_ptr = NULL; /* convenience ptr */
    dep->ready = FALSE;
    /* For the task, track the dependency to be use to do locality
     * preservation; by default, use first output dependency.  */
    if ( dep->locality )
        task->locality_preserving_dep = dep;
    else if ( (task->locality_preserving_dep == NULL) && ( dep->direction==OUTPUT || dep->direction==INOUT) )
        task->locality_preserving_dep = dep;
    return dep;
}

/* **************************************************************************** */
/**
 * Allocate and initialize a worker structure
 */
static Worker *worker_new(Quark *quark, int rank)
{
    Worker *worker = (Worker *) malloc(sizeof(Worker));
    assert(worker != NULL);
    worker->thread_id = pthread_self();
    worker->ready_list = malloc(sizeof(task_priority_tree_head_t));
    assert(worker->ready_list != NULL);
    RB_INIT( worker->ready_list );
    worker->ready_list_size = 0;
    pthread_mutex_init(&worker->ready_list_mutex, NULL);
    /* convenience pointer to the real args for the task  */
    worker->current_task_ptr = NULL;
    worker->quark_ptr = quark;
    worker->finalize = FALSE;
    worker->executing_task = FALSE;
    return worker;
}

/* **************************************************************************** */
/**
 * Cleanup and free worker data structures
 */
static void worker_delete(Worker * worker)
{
    task_priority_tree_node_t *node, *nxt;
    /* Destroy the workers priority queue, if there is still anything there */
    for ( node = RB_MIN( task_priority_tree_head_s, worker->ready_list ); node != NULL; node = nxt) {
        nxt = RB_NEXT( task_priority_tree_head_s, worker->ready_list, node );
        RB_REMOVE( task_priority_tree_head_s, worker->ready_list, node );
        free(node);
    }
    free( worker->ready_list );
    pthread_mutex_destroy(&worker->ready_list_mutex);
    free(worker);
}

/* **************************************************************************** */
/**
 * The task requires scratch workspace, which will be allocated if
 * needed.  This records the scratch requirements.
 */
static Scratch *scratch_new( void *arg_ptr, int arg_size, icl_list_t *task_args_list_node_ptr )
{
    Scratch *scratch = (Scratch *)malloc(sizeof(Scratch));
    assert(scratch != NULL);
    scratch->ptr = arg_ptr;
    scratch->size = arg_size;
    scratch->task_args_list_node_ptr = task_args_list_node_ptr;
    return(scratch);
}

/* **************************************************************************** */
/**
 * Allocate any needed scratch space;
 */
static void scratch_allocate( Task *task )
{
    icl_list_t *scr_node;
    for (scr_node = icl_list_first( task->scratch_list );
         scr_node != NULL && scr_node->data != NULL;
         scr_node = icl_list_next(task->scratch_list, scr_node)) {
        Scratch *scratch = (Scratch *)scr_node->data;
        if ( scratch->ptr == NULL ) {
            /* Since ptr is null, space is to be allocted and attached */
            assert( scratch->size > 0 );
            void *scratchspace = malloc( scratch->size );
            assert( scratchspace != NULL );
            *(void **)scratch->task_args_list_node_ptr->data = scratchspace;
        }
    }
}

/* **************************************************************************** */
/**
 * Deallocate any scratch space.
 */
static void scratch_deallocate( Task *task )
{
    icl_list_t *scr_node;
    for (scr_node = icl_list_first( task->scratch_list );
         scr_node != NULL && scr_node->data!=NULL;
         scr_node = icl_list_next(task->scratch_list, scr_node)) {
        Scratch *scratch = (Scratch *)scr_node->data;
        if ( scratch->ptr == NULL ) {
            /* If scratch had to be allocated, free it */
            free(*(void **)scratch->task_args_list_node_ptr->data);
        }
    }
}

/* **************************************************************************** */
/**
 * Called by the master thread.  This routine does not do thread
 * management, so it can be used with a larger libarary.  Allocate and
 * initialize the scheduler data stuctures for the master and
 * num_threads worker threads.
 *
 * @param[in] num_threads
 *          Number of threads to be used (1 master and rest compute workers).
 * @return
 *          Pointer to the QUARK scheduler data structure.
 * @ingroup QUARK
 */
Quark *QUARK_Setup(int num_threads)
{
    int i = 0;
    Quark *quark = (Quark *) malloc(sizeof(Quark));
    assert(quark != NULL);
    /* Used to tell master when to act as worker */
    int quark_unroll_tasks_per_thread = quark_getenv_int("QUARK_UNROLL_TASKS_PER_THREAD", 20);
    int quark_unroll_tasks = quark_getenv_int("QUARK_UNROLL_TASKS", quark_unroll_tasks_per_thread * num_threads);
    quark->war_dependencies_enable = quark_getenv_int("QUARK_WAR_DEPENDENCIES_ENABLE", 0);
    quark->queue_before_computing = quark_getenv_int("QUARK_QUEUE_BEFORE_COMPUTING", 0);
    quark->dot_dag_enable = quark_getenv_int("QUARK_DOT_DAG_ENABLE", 0);
    if ( quark->dot_dag_enable ) quark->queue_before_computing = 1;
    if ( quark->queue_before_computing==1 || quark_unroll_tasks==0 ) {
        quark->high_water_mark = (int)(INT_MAX - 1);
        quark->low_water_mark = (int)(quark->high_water_mark);
    } else {
        quark->low_water_mark = (int)(quark_unroll_tasks);
        quark->high_water_mark = (int)(quark->low_water_mark + quark->low_water_mark*0.25);
    }
    quark->num_queued_tasks = 0;
    pthread_cond_init( &quark->num_queued_tasks_cond, NULL );
    quark->num_threads = num_threads;
    quark->list_robin = 0;
    quark->start = FALSE;
    quark->all_tasks_queued = FALSE;
    quark->num_tasks = 0;
    quark->task_set = icl_hash_create( 0x1<<12, ullong_hash_function, ullong_key_compare );
    pthread_mutex_init( &quark->task_set_mutex, NULL );
    /* Define some function pointers that match a C++ interface */
    quark->rank = QUARK_Thread_Rank;
    /* Create hash table to hold addresses */
    quark->address_set = icl_hash_create( 0x01<<12, address_hash_function, address_key_compare);
    pthread_mutex_init(&quark->address_set_mutex, NULL);
    /* To handle completed tasks */
    quark->completed_tasks = malloc(sizeof(completed_tasks_head_t));
    assert ( quark->completed_tasks != NULL );
    TAILQ_INIT( quark->completed_tasks );
    pthread_mutex_init(&quark->completed_tasks_mutex, NULL);
    /* Setup workers */
    quark->worker = (Worker **) malloc(num_threads * sizeof(Worker *));
    assert(quark->worker != NULL);
    /* The structure for the 0th worker will be used by the master */
    quark->worker[0] = worker_new(quark, 0);
    quark->worker[0]->thread_id = pthread_self();
    if ( quark->dot_dag_enable ) {
        fopen(&dot_dag_file, DOT_DAG_FILENAME, "w"); /* global FILE variable */
        fprintf(dot_dag_file, "digraph G { size=\"10,7.5\"; center=1; orientation=portrait; \n");
        for (i=0; i<tasklevel_width_max_level; i++ )
            quark->tasklevel_width[i] = 0;
        pthread_mutex_init(&quark->dot_dag_mutex, NULL);
        /* fprintf(dot_dag_file, "%d [label=\"%d %d\",style=\"invis\"]\n", 0, 0, quark->tasklevel_width[i] ); */
        fprintf(dot_dag_file, "%d [style=\"invis\"]\n", 0);
    }
    /* Launch workers; first create the structures */
    for(i = 1; i < num_threads; i++)
        quark->worker[i] = worker_new(quark, i);
    /* Threads can start as soon as they want */
    quark->start = TRUE;
    return quark;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Allocate and initialize the scheduler
 * data stuctures and spawn worker threads.  Used when this scheduler
 * is to do all the thread management.
 *
 * @param[in] num_threads
 *          Number of threads to be used (1 master and rest compute workers).
 *          If num_threads < 1, first try environment variable QUARK_NUM_THREADS
 *          or use use num_threads = number of cores
 * @return
 *          Pointer to the QUARK data structure.
 * @ingroup QUARK
 */
Quark *QUARK_New(int num_threads)
{
    int i, nthrd;

    /* Init number of cores and topology */
    quark_topology_init();
    /* Get number of threads */
    if ( num_threads < 1 ) {
        nthrd = quark_get_numthreads();
        if ( nthrd == -1 ) {
            nthrd = 1;
        }
    }
    else
        nthrd = num_threads;

    /* Create scheduler data structures for master and workers */
    Quark *quark = QUARK_Setup(nthrd);
    /* Get binding informations */
    quark->coresbind = quark_get_affthreads();
    /* Setup thread attributes */
    pthread_attr_init(&quark->thread_attr);
    /* pthread_setconcurrency(quark->num_threads); */
    pthread_attr_setscope(&quark->thread_attr, PTHREAD_SCOPE_SYSTEM);
    /* Then start the threads, so that workers can scan the structures easily */
    for(i = 1; i < nthrd; i++) {
        int rc = pthread_create(&quark->worker[i]->thread_id, &quark->thread_attr, (void *(*)(void *))work_set_affinity_and_call_main_loop, quark->worker[i]);
        assert(rc == 0);
    }
    quark_setaffinity( quark->coresbind[0] );
    return quark;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Wait for all the tasks to be
 * completed, then return.  The worker tasks will NOT exit from their
 * work loop.
 *
 * @param[in,out] quark
 *         The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Barrier(Quark * quark)
{
    quark->all_tasks_queued = TRUE;
    while ( quark->num_tasks > 0 ) {
        process_completed_tasks(quark);
        work_main_loop( quark->worker[0] );
    }
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Wait for all the
 * tasks to be completed, then return.  The worker tasks will also
 * exit from their work loop at this time.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Waitall(Quark * quark)
{
    int i;
    QUARK_Barrier( quark );
    /* Tell each worker to exit the work_loop; master handles himself */
    for (i=1; i<quark->num_threads; i++)
        quark->worker[i]->finalize = TRUE;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Free all QUARK data structures, this
 * assumes that all usage of QUARK is completed.  This interface does
 * not manage, delete or close down the worker threads.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Free(Quark * quark)
{
    int i;
    QUARK_Waitall(quark);
    /* Write the level matching/forcing information */
    if ( quark->dot_dag_enable ) {
        for (i=1; i<tasklevel_width_max_level && quark->tasklevel_width[i]!=0; i++ ) {
            fprintf(dot_dag_file, "%d [label=\"%d:%d\"]\n", i, i, quark->tasklevel_width[i] );
            fprintf(dot_dag_file, "%d->%d [style=\"invis\"];\n", i-1, i );
        }
        fprintf(dot_dag_file, "} \n");
    }
    /* Destroy hash tables, workers and other data structures */
    for (i = 1; i < quark->num_threads; i++)
        worker_delete( quark->worker[i] );
    worker_delete( quark->worker[0] );
    if (quark->worker) free(quark->worker);
    if (quark->completed_tasks) free(quark->completed_tasks);
    icl_hash_destroy(quark->address_set, NULL, NULL);
    icl_hash_destroy(quark->task_set, NULL, NULL);
    if ( quark->dot_dag_enable ) {
        pthread_mutex_destroy(&quark->dot_dag_mutex);
        fclose(dot_dag_file);
    }
    pthread_mutex_destroy(&quark->address_set_mutex);
    free(quark);
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Wait for all tasks to complete, then
 * join/end the worker threads, and clean up all the data structures.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Delete(Quark * quark)
{
    void *exitcodep = NULL;
    int   i;

    QUARK_Waitall( quark );
    /* Wait for workers to quit and join threads */
    for (i = 1; i < quark->num_threads; i++)
        pthread_join(quark->worker[i]->thread_id, &exitcodep);
    pthread_attr_destroy( &quark->thread_attr );
    /* Destroy specific structures */
    if (quark->coresbind) free(quark->coresbind);
    quark_topology_finalize();
    /* Destroy hash tables, workers and other data structures */
    QUARK_Free( quark );
}

/* **************************************************************************** */
/**
 * Use the task_flags data structure to set various items in the task
 * (priority, lock_to_thread, color, labels, etc )
*/
Task *quark_set_task_flags_in_task_structure( Quark *quark, Task *task, Quark_Task_Flags *task_flags )
{
    if ( task_flags ) {
        if ( task_flags->task_priority ) task->priority = task_flags->task_priority;
        if ( task_flags->task_lock_to_thread >= 0 ) task->lock_to_thread = task_flags->task_lock_to_thread;
        if ( task_flags->task_color && quark->dot_dag_enable ) task->task_color = strdup(task_flags->task_color);
        if ( task_flags->task_label && quark->dot_dag_enable ) task->task_label = strdup(task_flags->task_label);
        if ( task_flags->task_sequence ) task->sequence = task_flags->task_sequence;
        if ( task_flags->task_thread_count > 1 ) task->task_thread_count = task_flags->task_thread_count;
    }
    return task;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  This is used in argument packing, to
 * create an initial task data structure.  Arguments can be packed
 * into this structure, and it can be submitted later.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] function
 *          The function (task) to be executed by the scheduler
 * @param[in] task_flags
 *          Flags to specify task behavior
 * @ingroup QUARK
 */
Task *QUARK_Task_Init(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags )
{
    Task *task = quark_task_new();
    task->function = function;
    quark_set_task_flags_in_task_structure( quark, task, task_flags );
    return task;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  This is used in argument packing, to
 * pack/add arguments to a task data structure.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in,out] task
 *          The task data struture to hold the arguments
 * @param[in] arg_size
 *          Size of the argument in bytes (0 cannot be used here)
 * @param[in] arg_ptr
 *          Pointer to data or argument
 * @param[in] arg_flags
 *          Flags indicating argument usage and various decorators
 *          INPUT, OUTPUT, INOUT, VALUE, NODEP, SCRATCH
 *          LOCALITY, ACCUMULATOR, GATHERV
 *          TASK_COLOR, TASK_LABEL (special decorators for VALUE)
 *          e.g., arg_flags    INPUT | LOCALITY | ACCUMULATOR
 *          e.g., arg_flags    VALUE | TASK_COLOR
 * @ingroup QUARK
 */
void QUARK_Task_Pack_Arg( Quark *quark, Task *task, int arg_size, void *arg_ptr, int arg_flags )
{
    icl_list_t *task_args_list_node_ptr=NULL;
    // extract information from the flags
    bool arg_locality = (bool) ((arg_flags & LOCALITY) != 0 );
    bool accumulator = (bool) ((arg_flags & ACCUMULATOR) != 0 );
    bool gatherv = (bool) ((arg_flags & GATHERV) != 0 );
    bool task_priority = (bool) ((arg_flags & TASK_PRIORITY) != 0 );
    bool task_lock_to_thread = (bool) ((arg_flags & TASK_LOCK_TO_THREAD) != 0 );
    bool task_thread_count = (bool) ((arg_flags & TASK_THREAD_COUNT) != 0 );
    bool task_color = (bool) ((arg_flags & TASK_COLOR) != 0 );
    bool task_label = (bool) ((arg_flags & TASK_LABEL) != 0 );
    bool task_sequence = (bool) ((arg_flags & TASK_SEQUENCE) != 0 );
    quark_direction_t arg_direction = (quark_direction_t) (arg_flags & DIRECTION_MASK);
    if (arg_direction == VALUE) {
        /* If argument is a value; Copy the contents to the argument buffer */
        if ( task_priority ) task->priority = *((int *)arg_ptr);
        else if ( task_lock_to_thread ) task->lock_to_thread = *((int *)arg_ptr);
        else if ( task_thread_count ) task->task_thread_count = *((int *)arg_ptr);
        else if ( task_sequence ) task->sequence = *((Quark_Sequence **)arg_ptr);
        else if ( task_color && quark->dot_dag_enable ) {
            if ( task->task_color && task->task_color!=quark_task_default_color) free(task->task_color);
            task->task_color = arg_dup(arg_ptr, arg_size);
        }
        else if ( task_label && quark->dot_dag_enable ) {
            if ( task->task_label && task->task_label!=quark_task_default_label) free(task->task_label);
            task->task_label = arg_dup(arg_ptr, arg_size) ;
        }
        else task_args_list_node_ptr = icl_list_append(task->args_list, arg_dup(arg_ptr, arg_size));
    } else {
        /* Else - argument is a pointer; Copy the pointer to the argument buffer - pass by reference */
        task_args_list_node_ptr = icl_list_append(task->args_list, arg_dup((char *) &arg_ptr, sizeof(char *)));
    }
    if ((arg_ptr != NULL) && ( arg_direction==INPUT || arg_direction==INOUT || arg_direction==OUTPUT )) {
        /* If argument is a dependency/slice, add dependency to task dependency list */
        Dependency *dep = dependency_new(arg_ptr, arg_size, arg_direction, arg_locality, task, accumulator, gatherv, task_args_list_node_ptr);
        icl_list_t *task_dependency_list_node_ptr = icl_list_append( task->dependency_list, dep );
        dep->task_dependency_list_node_ptr = task_dependency_list_node_ptr;
        task->num_dependencies++;
        task->num_dependencies_remaining++;
    }
    else if( arg_direction==SCRATCH ) {
        Scratch *scratch = scratch_new( arg_ptr, arg_size, task_args_list_node_ptr);
        icl_list_append( task->scratch_list, scratch );
    }
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Add a new task to the scheduler,
 * providing the data pointers, sizes, and dependency information.
 * This function provides the main user interface for the user to
 * write data-dependent algorithms.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in,out] task
 *          The packed task structure that already has all the
 *          arguments associated with the function
 * @ingroup QUARK
 */
unsigned long long QUARK_Insert_Task_Packed(Quark * quark, Task *task )
{
    unsigned long long taskid = task->taskid;
    /* Track sequence information if it is provided */
    if ( task->sequence ) {
/*         if ( task->sequence->status == QUARK_ERR ) { */
/*             task_delete( quark, task ); */
/*             return QUARK_ERR; */
/*         } else { */
/*             ll_list_node_t *entry = malloc(sizeof(ll_list_node_t)); */
/*             entry->val = task->taskid; */
/*             ll_list_head_t *headp = task->sequence->tasks_in_sequence; */
/*             pthread_mutex_lock_wrap( &task->sequence->sequence_mutex ); */
/*             LIST_INSERT_HEAD( headp, entry, entries ); */
/*             pthread_mutex_unlock_wrap( &task->sequence->sequence_mutex ); */
/*             /\* Keep pointer to task in sequence so it can be deleted when task completes *\/ */
/*             task->ptr_to_task_in_sequence = entry; */
/*             printf("sequence %p task %ld addto \n", task->sequence, task->taskid ); */
/*         } */
        /* TODO FIXME */
        if ( task->sequence->status == QUARK_ERR )
            task->function = NULL;
        ll_list_node_t *entry = malloc(sizeof(ll_list_node_t));
        entry->val = task->taskid;
        ll_list_head_t *headp = task->sequence->tasks_in_sequence;
        pthread_mutex_lock_wrap( &task->sequence->sequence_mutex );
        LIST_INSERT_HEAD( headp, entry, entries );
        pthread_mutex_unlock_wrap( &task->sequence->sequence_mutex );
        /* Keep pointer to task in sequence so it can be deleted when task completes */
        task->ptr_to_task_in_sequence = entry;
        //printf("sequence %p task %ld addto \n", task->sequence, task->taskid );

   }
    /* Insert the task in the address hash, locking access to the address set hash */
    pthread_mutex_lock_asn( &quark->address_set_mutex );
    /* For repeated usage of the scheduler, if tasks are being added repeatedly
     * then quark->finalize and quark->all_tasks_queued must be set false */
    quark->all_tasks_queued = FALSE;
    quark_insert_task_dependencies( quark, task );
    /* FIXME does this need to be protected */
    quark->num_tasks++;
    /* Save the task, indexed by its taskid */
    pthread_mutex_lock_wrap( &quark->task_set_mutex );
    icl_hash_insert( quark->task_set, &task->taskid, task );
    pthread_mutex_unlock_wrap( &quark->task_set_mutex );
    // Check if the task is ready
    quark_check_and_queue_ready_task( quark, task );
    pthread_mutex_unlock_asn(&quark->address_set_mutex);
    /* If conditions are right, master works; this will return when
     * num_tasks becomes less than low_water_mark */
    process_completed_tasks(quark);
    while (quark->num_tasks >= quark->high_water_mark) {
        work_main_loop(quark->worker[0]);
        process_completed_tasks(quark);
    }
    return taskid ;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Add a new task to the scheduler,
 * providing the data pointers, sizes, and dependency information.
 * This function provides the main user interface for the user to
 * write data-dependent algorithms.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] function
 *          The function (task) to be executed by the scheduler
 * @param[in] task_flags
 *          Flags to specify task behavior
 * @param[in] ...
 *          Triplets of the form, ending with 0 for arg_size.
 *            arg_size, arg_ptr, arg_flags where
 *          arg_size: int: Size of the argument in bytes (0 cannot be used here)
 *          arg_ptr: pointer: Pointer to data or argument
 *          arg_flags: int: Flags indicating argument usage and various decorators
 *            INPUT, OUTPUT, INOUT, VALUE, NODEP, SCRATCH
 *            LOCALITY, ACCUMULATOR, GATHERV
 *            TASK_COLOR, TASK_LABEL (special decorators for VALUE)
 *            e.g., arg_flags    INPUT | LOCALITY | ACCUMULATOR
 *            e.g., arg_flags    VALUE | TASK_COLOR
 * @return
 *          A long, long integer which can be used to refer to
 *          this task (e.g. for cancellation)
 * @ingroup QUARK
 */
unsigned long long QUARK_Insert_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...)
{
    va_list varg_list;
    int arg_size;
    unsigned long long taskid;

    Task *task = QUARK_Task_Init(quark, function, task_flags);

    va_start(varg_list, task_flags);
    // For each argument
    while( (arg_size = va_arg(varg_list, int)) != 0) {
        void *arg_ptr = va_arg(varg_list, void *);
        int arg_flags = va_arg(varg_list, int);
        QUARK_Task_Pack_Arg( quark, task, arg_size, arg_ptr, arg_flags );
    }
    va_end(varg_list);

    taskid = QUARK_Insert_Task_Packed( quark, task );

    return taskid ;
}

/* **************************************************************************** */
/**
 * Run this task in the current thread, at once, without scheduling.
 * This is an unsupported function that can be used by developers for
 * testing.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] function
 *          The function (task) to be executed by the scheduler
 * @param[in] task_flags
 *          Flags to specify task behavior
 * @param[in] ...
 *          Triplets of the form, ending with 0 for arg_size.
 *            arg_size, arg_ptr, arg_flags where
 *          arg_size: int: Size of the argument in bytes (0 cannot be used here)
 *          arg_ptr: pointer: Pointer to data or argument
 *          arg_flags: int: Flags indicating argument usage and various decorators
 *            INPUT, OUTPUT, INOUT, VALUE, NODEP, SCRATCH
 *            LOCALITY, ACCUMULATOR, GATHERV
 *            TASK_COLOR, TASK_LABEL (special decorators for VALUE)
 *            e.g., arg_flags    INPUT | LOCALITY | ACCUMULATOR
 *            e.g., arg_flags    VALUE | TASK_COLOR
 * @return
 *           -1 since the task is run at once and there is no need for a task handle.
 * @ingroup QUARK_Unsupported
 */
unsigned long long QUARK_Execute_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...)
{
    va_list varg_list;
    int arg_size;

    Task *task = QUARK_Task_Init(quark, function, task_flags);

    va_start(varg_list, task_flags);
    // For each argument
    while( (arg_size = va_arg(varg_list, int)) != 0) {
        void *arg_ptr = va_arg(varg_list, void *);
        int arg_flags = va_arg(varg_list, int);
        QUARK_Task_Pack_Arg( quark, task, arg_size, arg_ptr, arg_flags );
    }
    va_end(varg_list);

    int thread_rank = QUARK_Thread_Rank(quark);
    Worker *worker = quark->worker[thread_rank];
    if ( task->function == NULL ) {
        /* This can occur if the task is cancelled */
        task->status = CANCELLED;
    } else {
        /* Call the task */
        task->status = RUNNING;
        worker->current_task_ptr = task;
        scratch_allocate( task );
        task->function( quark );
        scratch_deallocate( task );
        worker->current_task_ptr = NULL;
        task->status = DONE;
    }

    /* Delete the task data structures */
    icl_list_destroy(task->args_list, free);
    icl_list_destroy(task->dependency_list, free);
    icl_list_destroy(task->scratch_list, free);
    pthread_mutex_destroy(&task->task_mutex);
    free(task);

    /* There is no real taskid to be returned, since the task has been deleted */
    return( -1 );
}

/* **************************************************************************** */
/**
 * Called by any thread.  Cancel a task that is in the scheduler.
 * This works by simply making the task a NULL task.  The scheduler
 * still processes all the standard dependencies for this task, but
 * when it is time to run the actual function, the scheduler does
 * nothing.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] taskid
 *          The taskid returned by a QUARK_Insert_Task
 * @return 1 on success.
 * @return -1 if the task cannot be found (may already be done and removed).
 * @return -2 if the task is aready running, done, or cancelled.
 * @ingroup QUARK
 */
int QUARK_Cancel_Task(Quark *quark, unsigned long long taskid)
{
    pthread_mutex_lock_wrap( &quark->task_set_mutex );
    Task *task = icl_hash_find( quark->task_set, &taskid );
    if ( task == NULL ) {
        pthread_mutex_unlock_wrap( &quark->task_set_mutex );
        return -1;
    }
    pthread_mutex_lock_wrap( &task->task_mutex );
    pthread_mutex_unlock_wrap( &quark->task_set_mutex );
    if ( task->status==RUNNING || task->status==DONE || task->status==CANCELLED ) {
        pthread_mutex_unlock_wrap( &task->task_mutex );
        return -2;
    }
    task->function = NULL;
    pthread_mutex_unlock_wrap( &task->task_mutex );
    return 1;
}

/* **************************************************************************** */
/**
 * Allocate and initialize address_set_node structure.  These are
 * inserted into the hash table.
 */
static Address_Set_Node *address_set_node_new( void* address, int size )
{
    Address_Set_Node *address_set_node = (Address_Set_Node *)malloc(sizeof(Address_Set_Node));
    assert( address_set_node != NULL );
    address_set_node->address = address;
    address_set_node->size = size;
    address_set_node->last_thread = -1;
    address_set_node->waiting_deps = icl_list_new();
    assert( address_set_node->waiting_deps != NULL );
    address_set_node->num_waiting_input = 0;
    address_set_node->num_waiting_output = 0;
    address_set_node->num_waiting_inout = 0;
    address_set_node->delete_data_at_address_when_node_is_deleted = FALSE;
    address_set_node->last_writer_taskid = 0;
    address_set_node->last_writer_tasklevel = 0;
    address_set_node->last_reader_or_writer_taskid = 0;
    address_set_node->last_reader_or_writer_tasklevel = 0;
    return address_set_node;
}

/* **************************************************************************** */
/**
 * Clean and free address set node structures.
 */
static void address_set_node_delete( Quark *quark, Address_Set_Node *address_set_node )
{
    /* Free data if it was allocted as a WAR data copy */
    if ( address_set_node->delete_data_at_address_when_node_is_deleted == TRUE ) {
        free( address_set_node->address );
    }
    /* Do not free this structure if we are generating DAGs.  The
     * structure contains information about the last task to write the
     * data used to make DAG edges */
    if ( quark->dot_dag_enable )
        return;

    /* Remove any data structures in the waiting_deps list */
    if ( address_set_node->waiting_deps != NULL )
        icl_list_destroy( address_set_node->waiting_deps, NULL );
    /* Delete and free the hash table entry if this was NOT a WAR create entry */
    icl_hash_delete( quark->address_set, address_set_node->address, NULL, NULL );
    /* Remove the data structure */
    free( address_set_node );
}

/* **************************************************************************** */
/**
 * Queue ready tasks on a worker node, either using locality
 * information or a round robin scheme.  The address_set_mutex should
 * be set when calling this, since we touch the task data structure
 * (task->status) and update the quark->num_queued_tasks.
 */
static void quark_check_and_queue_ready_task( Quark *quark, Task *task )
{
    int worker_thread_id = -1;
    Worker *worker = NULL;
    int assigned_thread_count = 0;

    if ( task->num_dependencies_remaining > 0 || task->status == QUEUED || task->status == RUNNING || task->status == DONE) return;
    task->status = QUEUED;
    /* Assign task to thread.  Locked tasks get sent to appropriate
     * thread.  Locality tasks should have be correctly placed.  Tasks
     * without either should have the original round robin thread
     * assignment */
    if ( task->lock_to_thread >= 0) {
        worker_thread_id = task->lock_to_thread % quark->num_threads;
    } else if ( task->locality_preserving_dep != NULL ) {
        int last_thread = task->locality_preserving_dep->address_set_node_ptr->last_thread;
        if ( last_thread >= 0 ) worker_thread_id = last_thread;
    }
    if ( worker_thread_id < 0 ) worker_thread_id = quark_revolve_robin(quark);

    /* Handle tasks that need multiple threads */
    while ( assigned_thread_count < task->task_thread_count) {

        worker = quark->worker[worker_thread_id];
        /* Create a new entry for the ready list */
        task_priority_tree_node_t *new_task_tree_node = malloc(sizeof(task_priority_tree_node_t));
        assert( new_task_tree_node != NULL );
        new_task_tree_node->priority = task->priority;
        new_task_tree_node->task = task;
        /* Insert new entry into the ready list */
        pthread_mutex_lock_ready_list(&worker->ready_list_mutex);
        RB_INSERT( task_priority_tree_head_s, worker->ready_list, new_task_tree_node );
        worker->ready_list_size++;
        pthread_mutex_unlock_ready_list(&worker->ready_list_mutex);
        pthread_cond_broadcast( &quark->num_queued_tasks_cond );
        quark->num_queued_tasks++;

        assigned_thread_count++;
        /* TODO Abort when too many threads requested */
        if ( assigned_thread_count < task->task_thread_count )
            worker_thread_id = (worker_thread_id+1) % quark->num_threads;
    }
}

/* **************************************************************************** */
/**
 * Routine to avoid false (WAR write-after-read) dependencies by
 * making copies of the data.  Check if there are suffient INPUTS in
 * the beginning of a address dependency followed by a OUTPUT or an
 * INOUT (data<-RRRRW).  If so, make a copy of the data, adjust the
 * pointers of the read dependencies to point to the new copy
 * (copy<-RRRR and data<-W) and send to workers if the tasks are
 * ready.  The copy can be automacally freed when all the reads are
 * done.  The write can proceed at once.  The address_set_mutex is
 * already locked when this is called.
 */
void quark_avoid_war_dependencies( Quark *quark, Address_Set_Node *asn_old, Task *parent_task )
{
    /* Figure out if there are enough input dependencies to make this worthwhile */
    int count_initial_input_deps = 0;
    bool output_dep_reached = FALSE;
    double avg_queued_tasks_per_thread = (double)quark->num_queued_tasks/(double)quark->num_threads;
    double avg_tasks_per_thread = (double)quark->num_tasks/(double)quark->num_threads;
    int min_input_deps;
    icl_list_t *dep_node_old;

    /* Quick return if this is not enabled */
    if ( !quark->war_dependencies_enable ) return;

    /* TODO This stuff is still under development.... */
    if ( avg_queued_tasks_per_thread < 0.4 ) min_input_deps = 1;
    else if ( avg_queued_tasks_per_thread < 0.75 ) min_input_deps = 6;
    else if ( avg_queued_tasks_per_thread < 0.90 ) min_input_deps = 7;
    else if ( avg_queued_tasks_per_thread < 1.20 ) min_input_deps = 10;
    else if ( avg_queued_tasks_per_thread > 1.80 ) min_input_deps = 2000;
    else if ( avg_tasks_per_thread < (double)quark->low_water_mark/(double)quark->num_threads/2 ) min_input_deps = 2000;
    else min_input_deps = (int)(7 + 27 * avg_queued_tasks_per_thread);

    /* Override computed value using environment variable */
    min_input_deps = quark_getenv_int( "QUARK_AVOID_WAR_WHEN_NUM_WAITING_READS", min_input_deps );

    /* Shortcut return if there are not enough input tasks */
    if ( asn_old->num_waiting_input < min_input_deps ) return;

    /* Scan thru initial deps, make sure they are inputs and that there
     * are enough of them to make data copying worthwhile */
    for (dep_node_old=icl_list_first(asn_old->waiting_deps);
         dep_node_old!=NULL;
         dep_node_old=icl_list_next(asn_old->waiting_deps, dep_node_old)) {
        Dependency *dep = (Dependency *)dep_node_old->data;
        Task *task = dep->task;
        if ( dep->direction==INPUT && task->status==NOTREADY  ) {
            count_initial_input_deps++;
        } else if ( (dep->direction==OUTPUT || dep->direction==INOUT) && task->status!=DONE ) {
            output_dep_reached = TRUE;
            break;
        }
    }

    /* if ( count_initial_input_deps>=quark->min_input_deps_to_avoid_war_dependencies && output_dep_reached ) { */
    if ( count_initial_input_deps>=min_input_deps && output_dep_reached ) {
        icl_list_t *dep_node_asn_old;
        Address_Set_Node *asn_new;
        /* Allocate and copy data */
        void *datacopy = malloc( asn_old->size );
        assert(datacopy!=NULL);
        /* TODO track the allocated memory in datacopies */
        /* quark->mem_allocated_to_war_dependency_data += asn_old->size; */
        memcpy( datacopy, asn_old->address, asn_old->size );
        /* Create address set node, attach to hash, and set it to clean up when done */
        asn_new = address_set_node_new( datacopy, asn_old->size );
        asn_new->delete_data_at_address_when_node_is_deleted = TRUE;
        icl_hash_insert( quark->address_set, asn_new->address, asn_new );
        /* Update task dependences to point to this new data */
        /* Grab input deps from the old list, copy to new list, delete, then repeat */
        for ( dep_node_asn_old=icl_list_first(asn_old->waiting_deps);
              dep_node_asn_old!=NULL;  ) {
            icl_list_t *dep_node_asn_old_to_be_deleted = NULL;
            Dependency *dep = (Dependency *)dep_node_asn_old->data;
            Task *task = dep->task;
            if ( dep->direction==INPUT && task->status==NOTREADY ) {
                dep_node_asn_old_to_be_deleted = dep_node_asn_old;
                icl_list_t *dep_node_new = icl_list_append( asn_new->waiting_deps, dep );
                asn_new->num_waiting_input++;
                /* In the args list, set the arg pointer to the new datacopy address */
                *(void **)dep->task_args_list_node_ptr->data = datacopy;
                dep->address = asn_new->address;
                dep->address_set_node_ptr = asn_new;
                dep->address_set_waiting_deps_node_ptr = dep_node_new;
                if (dep->ready == FALSE) { /* dep->ready will always be FALSE */
                    dep->ready = TRUE;
                    dot_dag_print_edge( parent_task->taskid, task->taskid, DEPCOLOR );
                    dot_dag_level_update( parent_task->tasklevel, task->tasklevel, quark );
                    task->num_dependencies_remaining--;
                    quark_check_and_queue_ready_task( quark, task );
                }
            } else if ( (dep->direction==OUTPUT || dep->direction==INOUT) && task->status!=DONE ) {
                /* Once we return from this routine, this dep dependency will be processed */
                break;
            }
            dep_node_asn_old = icl_list_next(asn_old->waiting_deps, dep_node_asn_old);
            if (dep_node_asn_old_to_be_deleted!=NULL) {
                icl_list_delete(asn_old->waiting_deps, dep_node_asn_old_to_be_deleted, NULL);
            }
        }
    }
}

/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Sweeps through a sequence of GATHERV dependencies from the
 * beginning, and enables them all. Assumes address_set_mutex is
 * locked.
 */
static void address_set_node_initial_gatherv_check_and_launch(Quark *quark, Address_Set_Node *address_set_node, Dependency *completed_dep, int worker_rank)
{
    icl_list_t *next_dep_node;
    Task *completed_task = completed_dep->task;
    for ( next_dep_node=icl_list_first(address_set_node->waiting_deps);
          next_dep_node!=NULL && next_dep_node->data != NULL;
          next_dep_node=icl_list_next(address_set_node->waiting_deps, next_dep_node) ) {
        Dependency *next_dep = (Dependency *)next_dep_node->data;
        /* Break when we run out of GATHERV output dependencies */
        if ( next_dep->gatherv==FALSE ) break;
        if ( next_dep->direction!=OUTPUT && next_dep->direction!=INOUT ) break;
        Task *next_task = next_dep->task;
        /* Update next_dep ready status */
        if ( next_dep->ready == FALSE ) {
            /* Record the locality information with the task data structure */
            //if ( next_dep->locality ) next_task->locality_preserving_dep = worker_rank;
            /* Mark the next dependency as ready since we have GATHERV flag */
            next_dep->ready = TRUE;
            dot_dag_print_edge( completed_task->taskid, next_task->taskid, GATHERVDEPCOLOR );
            dot_dag_level_update( completed_task->tasklevel, next_task->tasklevel, quark );
            next_task->num_dependencies_remaining--;
            /* If the dep status became true check related task, and put onto ready queues */
            quark_check_and_queue_ready_task( quark, next_task );
        }

    }
}

/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Sweeps through a sequence of ACCUMULATOR tasks from the
 * beginning and prepends one at the beginning if only one (chained)
 * dependency remaining. This does not actually lauch the prepended
 * task, it depends on another function to do that. Assumes
 * address_set_mutex is locked.
 */
static void address_set_node_accumulator_find_prepend(Quark *quark, Address_Set_Node *address_set_node)
{
    icl_list_t *dep_node = NULL;
    icl_list_t *first_dep_node = NULL;
    icl_list_t *first_ready_dep_node = NULL;
    icl_list_t *last_ready_dep_node = NULL;
    icl_list_t *last_dep_node = NULL;
    icl_list_t *swap_node = NULL;
    int acc_dep_count = 0;

    /* FOR each ACCUMULATOR task waiting at the beginning of address_set_node  */
    for (dep_node = icl_list_first(address_set_node->waiting_deps);
         dep_node != NULL;
         dep_node = icl_list_next( address_set_node->waiting_deps, dep_node )) {
        Dependency *dependency = (Dependency *)dep_node->data;
        /* IF not an ACCUMULATOR dependency - break */
        if (dependency->accumulator == FALSE) break;
        Task *task = dependency->task;
        /* Scan through list keeping first, first_ready, last_ready, last */
        if (first_dep_node==NULL) first_dep_node = dep_node;
        if ( task->num_dependencies_remaining==1 ) {
            if (first_ready_dep_node==NULL) first_ready_dep_node = dep_node;
            last_ready_dep_node = dep_node;
        }
        last_dep_node = dep_node; /* TODO */
        acc_dep_count++;
    }

    /* Choose and move chosen ready node to the front of the list */
    /* Heuristic: Flip-flop between first-ready and last-ready.
     * Tested (always first, always last, flip-flop first/last) but
     * there was always a bad scenario.  If perfect loop orders are
     * provided (e.g. Choleky inversion test) then this will not make
     * performance worse.  If bad loops are provided, this will
     * improve performance, though not to the point of perfect
     * loops.  */
    if (acc_dep_count % 2 == 0 ) {
        if ( last_ready_dep_node!=NULL ) swap_node = last_ready_dep_node;
    } else {
        if ( first_ready_dep_node != NULL ) swap_node = first_ready_dep_node;
    }
    if ( swap_node != NULL ) {
        Dependency *dependency = (Dependency *)swap_node->data;
        /* Move to front of the address_set_node waiting_deps list (if not already there) */
        if ( swap_node!=icl_list_first(address_set_node->waiting_deps) ) {
                icl_list_t *tmp_swap_node = icl_list_prepend( address_set_node->waiting_deps, dependency );
                dependency->address_set_waiting_deps_node_ptr = tmp_swap_node;
                icl_list_delete( address_set_node->waiting_deps, swap_node, NULL );
        }
        /* Lock the dependency in place by setting ACC to false now */
        dependency->accumulator = FALSE;
    }
}


/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Sweeps through a sequence of initial INPUT dependencies on
 * an address, and launches any that are ready to go. Assumes
 * address_set_mutex is locked.
 */
static void address_set_node_initial_input_check_and_launch(Quark *quark, Address_Set_Node *address_set_node, Dependency *completed_dep, int worker_rank)
{
    icl_list_t *next_dep_node;
    Task *completed_task = completed_dep->task;
    for ( next_dep_node=icl_list_first(address_set_node->waiting_deps);
          next_dep_node!=NULL && next_dep_node->data != NULL;
          next_dep_node=icl_list_next(address_set_node->waiting_deps, next_dep_node) ) {
        Dependency *next_dep = (Dependency *)next_dep_node->data;
        Task *next_task = next_dep->task;
        /* Break when we hit an output dependency */
        if ( (next_dep->direction==OUTPUT || next_dep->direction==INOUT) ) {
            if ( completed_dep->direction == INPUT ) {
                /* Print DAG connections for antidependencies */
                dot_dag_print_edge( completed_task->taskid, next_task->taskid, ANTIDEPCOLOR );
                dot_dag_level_update( completed_task->tasklevel, next_task->tasklevel, quark );
            }
            break;
        }
        /* Update next_dep ready status; this logic assumes the breaks at the bottom */
        if ( next_dep->direction==INPUT && next_dep->ready == FALSE ) {
            /* Record the locality information with the task data structure */
            //if ( next_dep->locality ) next_task->locality_thread_id = worker_rank;
            /* If next_dep is INPUT, mark the next dependency as ready */
            next_dep->ready = TRUE;
            /* Only OUTPUT->INPUT edges get here */
            dot_dag_print_edge( completed_task->taskid, next_task->taskid, DEPCOLOR );
            dot_dag_level_update( completed_task->tasklevel, next_task->tasklevel, quark );
            next_task->num_dependencies_remaining--;
            /* If the dep status became true check related task, and put onto ready queues */
            quark_check_and_queue_ready_task( quark, next_task );
        }

        /* if we are generating the DAG, keep looping till an output
         * dependency (in order to print all WAR edges) */
        if (! quark->dot_dag_enable ) {
            /* If current original dependency (dep) was INPUT, we only need to
             * activate next INPUT/OUTPUT/INOUT dep, others should already be
             * handled; if original dep was OUTPUT/INOUT, need to keep
             * going till next OUTPUT/INOUT */
            if ( completed_dep->direction == INPUT ) break;
        }
    }
}


/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Checks any initial OUTPUT/INOUT dependencies on an address,
 * and launches any tasks that are ready to go. Assumes
 * address_set_mutex is locked.
 */
static void address_set_node_initial_output_check_and_launch(Quark *quark, Address_Set_Node *address_set_node, Dependency *completed_dep, int worker_rank)
{
    icl_list_t *next_dep_node;
    next_dep_node = icl_list_first(address_set_node->waiting_deps);
    if ( next_dep_node!=NULL && next_dep_node->data!=NULL ) {
        Dependency *next_dep = (Dependency *)next_dep_node->data;
        Task *next_task = next_dep->task;
        if ( (next_dep->direction==OUTPUT || next_dep->direction==INOUT) ) {
            /* Process OUTPUT next_deps, if at beginning of address_set_list waiting_deps starts  */
            if ( next_dep->ready == FALSE ) {
                /* Record the locality information with the task data structure */
                //if ( next_dep->locality ) next_task->locality_thread_id = worker_rank;
                /* If next_dep is output, mark the next dep as ready only if it is at the front */
                next_dep->ready = TRUE;
                Task *completed_task = completed_dep->task;
                if ( completed_dep->direction==OUTPUT || completed_dep->direction==INOUT )
                    dot_dag_print_edge( completed_task->taskid, next_task->taskid, DEPCOLOR );
                /*  else                      */ /* Handled in initial_input_check_and_launch */
                /*     dot_dag_print_edge( completed_task->taskid, next_task->taskid, ANTIDEPCOLOR ); */
                dot_dag_level_update( completed_task->tasklevel, next_task->tasklevel, quark );
                next_task->num_dependencies_remaining--;
                quark_check_and_queue_ready_task( quark, next_task );
            }
        }
    }
}

/* **************************************************************************** */
/**
 * Called by the master insert task dependencies into the hash table.
 * Any tasks that are ready to run are queued.  The address_set_mutex
 * must be locked before calling this routine.
 */
static void quark_insert_task_dependencies(Quark * quark, Task * task)
{
    icl_list_t *task_dep_p = NULL; /* task dependency list pointer */

    /* For each task dependency list pointer */
    for (task_dep_p = icl_list_first(task->dependency_list);
         task_dep_p != NULL;
         task_dep_p = icl_list_next(task->dependency_list, task_dep_p)) {
        Dependency *dep = (Dependency *) task_dep_p->data;
        /* Lookup address in address_set hash */
        Address_Set_Node *address_set_node = (Address_Set_Node *)icl_hash_find( quark->address_set, dep->address );
        /* If not found, create a new address set node and add it to the hash */
        if ( address_set_node == NULL ) {
            address_set_node = address_set_node_new( dep->address, dep->size );
            icl_hash_insert( quark->address_set, address_set_node->address, address_set_node );
        }
        /* Convenience shortcut pointer so that we don't have to hash again */
        dep->address_set_node_ptr = address_set_node;
        /* Add the dependency to the list of waiting dependencies on this address set node */
        icl_list_t *curr_dep_node = icl_list_append( address_set_node->waiting_deps, dep );
        /* Convenience shortcut pointer so we don't have to scan the waiting dependencies */
        dep->address_set_waiting_deps_node_ptr = curr_dep_node;
        /* Track num of waiting input, output and inout to be used to check false dependency resolution */
        if (dep->direction == INPUT) address_set_node->num_waiting_input++;
        else if (dep->direction == OUTPUT) address_set_node->num_waiting_output++;
        else if (dep->direction == INOUT) address_set_node->num_waiting_inout++;

        /* Handle the case that the a single task make multiple dependencies on the same data address */
        /* e.g. func( A11:IN, A11:INOUT, A11:OUT, A11:IN, A22:OUT )  */
        icl_list_t *prev_dep_node = icl_list_prev( address_set_node->waiting_deps, curr_dep_node);
        if ( prev_dep_node != NULL ) {
            Dependency *prev_dep = (Dependency *)prev_dep_node->data;
            Task *prev_task = prev_dep->task;
            if ( prev_task->taskid == task->taskid ) {
                /* The curr dependency will updated using the ordering INPUT < OUTPUT < INOUT  */
                /* When the scheduler checks the front of the dependency list, it will find the correct dep setting */
                dep->direction = (dep->direction > prev_dep->direction ? INOUT : prev_dep->direction );
                if ( prev_dep->ready == FALSE ) {
                    prev_dep->ready = TRUE;
                    task->num_dependencies_remaining--;
                }
                /* Remove the redundent dependency from waiting deps and from the task */
                icl_list_delete( address_set_node->waiting_deps, prev_dep_node, NULL );
                icl_list_delete( task->dependency_list, prev_dep->task_dependency_list_node_ptr, NULL );
                /* Update the prev_dep_node ptr since it has changed */
                prev_dep_node = icl_list_prev( address_set_node->waiting_deps, curr_dep_node);
            }
        }

        /* This will avoid WAR dependencies if possible: if enabled, and
         * the current dependency is a write, and there were only reads
         * earlier (input>1, output+inout=1) */
        if ( ((dep->direction==OUTPUT || dep->direction==INOUT)) &&
             ((address_set_node->num_waiting_output + address_set_node->num_waiting_inout) == 1) ) {
            quark_avoid_war_dependencies( quark, address_set_node, task );
        }

        /* The following code decides whether the dep is ready or not */
        if ( dep->direction==INOUT || dep->direction==OUTPUT ) {
            /* If output, and previous dep exists, then ready=false */
            if ( prev_dep_node != NULL ) {
                dep->ready = FALSE;
            } else {
                dep->ready = TRUE;
                dot_dag_print_edge( address_set_node->last_reader_or_writer_taskid, task->taskid, DEPCOLOR );
                dot_dag_level_update( address_set_node->last_reader_or_writer_tasklevel, task->tasklevel, quark );
                task->num_dependencies_remaining--;
            }
        } else if ( dep->direction == INPUT ) {
            if ( prev_dep_node != NULL ) {
                /* If input, and previous dep is a read that is ready, then ready=true */
                Dependency *prev_dep = (Dependency *)prev_dep_node->data;
                if ( prev_dep->direction==INPUT && prev_dep->ready==TRUE ) {
                    dep->ready = TRUE;
                    dot_dag_print_edge( address_set_node->last_writer_taskid, task->taskid, DEPCOLOR );
                    dot_dag_level_update( address_set_node->last_writer_tasklevel, task->tasklevel, quark );
                    task->num_dependencies_remaining--;
                } else {
                    dep->ready = FALSE;
                }
            } else {
                /* Input, but no previous node (is first), so ready   */
                dep->ready = TRUE;
                dot_dag_print_edge( address_set_node->last_writer_taskid, task->taskid, DEPCOLOR );
                dot_dag_level_update( address_set_node->last_writer_tasklevel, task->tasklevel, quark );
                task->num_dependencies_remaining--;
            }
        }
    }
}


/* **************************************************************************** */
/**
 * This function is called by a thread when it wants to start working.
 * This is used in a system that does its own thread management, so
 * each worker thread in that system must call this routine to get the
 * worker to participate in computation.
 *
 * @param[in,out] quark
 *          The main data structure.
 * @param[in] thread_rank
 *          The rank of the thread.
 * @ingroup QUARK
 */
void QUARK_Worker_Loop(Quark *quark, int thread_rank)
{
    quark->worker[thread_rank]->thread_id = pthread_self();
    work_main_loop( quark->worker[thread_rank] );
}


/* **************************************************************************** */
/**
 * Called when spawning the worker thread to set affinity to specific
 * core and then call the main work loop.  This function is used
 * internally, when the scheduler spawns and manages the threads.  If
 * an external driver is using the scheduler (e.g. PLASMA) then it
 * does the thread management and any affinity must be set in the
 * external driver.
 */
static void work_set_affinity_and_call_main_loop(Worker *worker)
{
    Quark *quark = worker->quark_ptr;
    int thread_rank = QUARK_Thread_Rank(quark);
    quark_setaffinity( quark->coresbind[thread_rank] ) ;
    work_main_loop( quark->worker[thread_rank] );
    return;
}

/* **************************************************************************** */
/**
 * Called by the workers (and master) to continue executing tasks
 * until some exit condition is reached.
 */
static void work_main_loop(Worker *worker)
{
    Quark *quark = worker->quark_ptr;
    Worker *worker_victim = NULL;
    task_priority_tree_node_t *task_priority_tree_node = NULL;
    Task *task = NULL;
    int ready_list_victim = -1;

    /* Busy wait while not ready */
    do {} while ( !quark->start );
    int worker_rank = QUARK_Thread_Rank(quark);

    /* Queue all tasks before running; this line for debugging use */
    /* while ( !quark->all_tasks_queued ) { if (worker_rank==0) return; else {} } */
    if ( quark->queue_before_computing )
        while ( !quark->all_tasks_queued ) { if (worker_rank==0) return; else {} }
    /* Master never does work; this line for debugging use  */
    /* if (worker_rank == 0) return; */

    while ( !worker->finalize ) {
        /* Repeatedly try to find a task, first trying my own ready list,
         * then trying to steal from someone else */
        task = NULL;
        ready_list_victim = worker_rank;
        /* Loop while looking for tasks */
        while ( task==NULL && !worker->finalize ) {

            /* Process all completed tasks before doing work */
            if ( worker_rank==0 || worker_rank%10==1 ) process_completed_tasks(quark);

            worker_victim = quark->worker[ready_list_victim];
            task_priority_tree_node = NULL;
            assert ( worker_victim->ready_list_size >= 0 );
            if ( worker_victim->ready_list_size != 0 ) {
                /* Only lock if there is likely to be an item in the ready list */
                if ( pthread_mutex_trylock_ready_list( &worker_victim->ready_list_mutex ) == 0) {
                    /* if (pthread_mutex_lock_ready_list(&worker_victim->ready_list_mutex)==0) { */
                    /* Check front of my own queue, back of everyone else's queue */
                    if ( worker_rank == ready_list_victim )
                        task_priority_tree_node = RB_MIN( task_priority_tree_head_s, worker_victim->ready_list );
                    else if ( worker_rank!=ready_list_victim && worker_victim->executing_task==TRUE )
                        task_priority_tree_node = RB_MAX( task_priority_tree_head_s, worker_victim->ready_list );
                    else
                        task_priority_tree_node = NULL;
                    /* Access task, checking to make sure it is not pinned to a thread */
                    if ( task_priority_tree_node != NULL ) {
                        task = task_priority_tree_node->task;
                        /* If task should be locked to a thread, and this is not that thread, set task to NULL and continue */
                        if ( task->lock_to_thread>=0 && task->lock_to_thread!=worker_rank) {
                            task = NULL;
                        } else {
                            /* If task found, remove it from the ready list */
                            RB_REMOVE( task_priority_tree_head_s, worker_victim->ready_list, task_priority_tree_node );
                            free( task_priority_tree_node );
                            worker_victim->ready_list_size--;
                        }
                    }
                    pthread_mutex_unlock_ready_list( &worker_victim->ready_list_mutex );
                }
            }
            /* If no task found */
            if (task == NULL) {
                /* Choose the next victim queue */
                ready_list_victim = (ready_list_victim + 1) % quark->num_threads;
                /* Break for master when a scan of all queues is finished and no tasks were found */
                if ( worker_rank==0 && ready_list_victim==0 ) break;
                /* If there are no tasks, wait for a task to be introduced, then check own queue first */
                if ( quark->num_queued_tasks==0 && !worker->finalize && worker_rank!=0 )  {
                    do { assert( quark->num_queued_tasks >= 0); } while ( quark->num_queued_tasks==0 && !worker->finalize ) ;
                    ready_list_victim = worker_rank;
                }
            }
        }
        /* EXECUTE THE TASK IF FOUND */
        if ( task!=NULL ) {
            //if ( quark->num_tasks != 1 ) { printf("quark->num_tasks %d %d %d\n", quark->num_tasks, quark->low_water_mark, quark->high_water_mark ); abort(); }
            pthread_mutex_lock_wrap( &task->task_mutex );
            if ( task->function == NULL ) {
                /* This can occur if the task is cancelled */
                task->status = CANCELLED;
                pthread_mutex_unlock_wrap( &task->task_mutex );
            } else {
                /* Call the task */
                worker->executing_task = TRUE;
                task->status = RUNNING;
                pthread_mutex_unlock_wrap( &task->task_mutex );
                scratch_allocate( task );
                worker->current_task_ptr = task;
                task->function( quark );
                scratch_deallocate( task );
                task->status = DONE;
                worker->executing_task = FALSE;
            }
            /* Remove the task from the address hash */
            /* Original solution */
            //pthread_mutex_lock_asn(&quark->address_set_mutex);
            //worker_remove_completed_task_and_check_for_ready(quark, task, worker_rank);
            //pthread_mutex_unlock_asn(&quark->address_set_mutex);
            /* New version */
            worker_remove_completed_task_enqueue_for_later_processing(quark, task, worker_rank);
        }
        /* Break if master */
        if ( worker_rank==0 && ready_list_victim==0 ) break;
    }
    /* Worker has exited loop; ready for next time this worker is activated */
    worker->finalize = FALSE;
}


/* **************************************************************************** */
/**
 * Called by the control program.  Creates a new sequence data
 * structure and returns it.  This can be used to put a sequence of
 * tasks into a group and cancel that group if an error condition
 * occurs.
 *
 * @param[in],out quark
 *          Pointer to the scheduler data structure
 * @return Pointer to the newly created sequence structure.
 * @ingroup QUARK
 */
Quark_Sequence *QUARK_Sequence_Create( Quark *quark )
{
    Quark_Sequence *sequence = malloc(sizeof(Quark_Sequence));
    assert( sequence != NULL );
    sequence->status = QUARK_SUCCESS;
    pthread_mutex_init( &sequence->sequence_mutex, NULL );
    ll_list_head_t *head = malloc(sizeof(ll_list_head_t));
    assert ( head != NULL );
    LIST_INIT(head);
    sequence->tasks_in_sequence = head;
    return sequence;
}

/* **************************************************************************** */
/**
 * Can be called by any thread.  Cancels all the remaining tasks in a
 * sequence using QUARK_Cancel_Task and changes the state so that
 * future tasks belonging to that sequence are ignored.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in,out] sequence
 *          Pointer to a sequence data structure
 * @return 0 (QUARK_SUCCESS) on success
 * @return -1 (QUARK_ERR) on failure
 * @ingroup QUARK
 */
int QUARK_Sequence_Cancel( Quark *quark, Quark_Sequence *sequence )
{
    int retval;
    if ( quark==NULL || sequence==NULL ) return QUARK_ERR;
    pthread_mutex_lock_wrap( &sequence->sequence_mutex );
    if ( sequence->status != QUARK_SUCCESS ) {
        /* sequence already cancelled */
        retval = QUARK_SUCCESS;
    } else {
        sequence->status = QUARK_ERR;
        ll_list_node_t *np, *np_temp;
        LIST_FOREACH_SAFE( np, sequence->tasks_in_sequence, entries, np_temp ) {
            long long int taskid = np->val;
            /* Find taskid, make function NULL */
            QUARK_Cancel_Task( quark, taskid );
            /* Task node is removed from sequence when it finishes and is
             * deleted; or when sequence is destroyed */
        }
        retval = QUARK_SUCCESS;
    }
    pthread_mutex_unlock_wrap( &sequence->sequence_mutex );
    return retval;
}

/* **************************************************************************** */
/**
 * Called by the control program.  Cancels all the remaining tasks in
 * a sequence using QUARK_Cancel_Task and deletes the sequence data
 * structure.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in,out] sequence
 *          Pointer to a sequence data structure
 * @return A NULL pointer; which can be used to reset the sequence structure
 * @ingroup QUARK
 */
Quark_Sequence *QUARK_Sequence_Destroy( Quark *quark, Quark_Sequence *sequence )
{
    if ( quark==NULL || sequence==NULL) return NULL;
    //printf("QUARK_Sequence_Destroy %p status %d\n", sequence, sequence->status);
    pthread_mutex_lock_wrap( &sequence->sequence_mutex );
    ll_list_node_t *np, *np_temp;
    ll_list_head_t *head = sequence->tasks_in_sequence;
    LIST_FOREACH_SAFE( np, head, entries, np_temp ) {
        long long int taskid = np->val;
        QUARK_Cancel_Task( quark, taskid );
    }
    pthread_mutex_unlock_wrap( &sequence->sequence_mutex );
    QUARK_Sequence_Wait( quark, sequence );
    pthread_mutex_lock_wrap( &sequence->sequence_mutex );
    LIST_FOREACH_SAFE( np, head, entries, np_temp ) {
        LIST_REMOVE( np, entries );
        free( np );
    }
    pthread_mutex_unlock_wrap( &sequence->sequence_mutex );
    free( head );
    head = NULL;
    pthread_mutex_destroy( &sequence->sequence_mutex );
    free( sequence );
    sequence = NULL;
    return sequence;
}

/* **************************************************************************** */
/**
 * Called by the control program.  Returns when all the tasks in a
 * sequence have completed.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in,out] sequence
 *          Pointer to a sequence structure
 * @return  0 on success
 * @return  -1 on failure
 * @ingroup QUARK
 */
int QUARK_Sequence_Wait( Quark *quark, Quark_Sequence *sequence )
{
    if ( quark==NULL || sequence==NULL) return QUARK_ERR;
    int myrank = QUARK_Thread_Rank( quark );
    while ( !LIST_EMPTY( sequence->tasks_in_sequence ) ) {
        process_completed_tasks( quark );
        work_main_loop( quark->worker[myrank] );
    }
    return QUARK_SUCCESS;
}


/* **************************************************************************** */
/**
 * For the current thread, in the current task being executed, return
 * the task's sequence value.  This is the value provided when the
 * task was Task_Inserted into a sequence.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @return Pointer to sequence data structure
 * @ingroup QUARK
 */
Quark_Sequence *QUARK_Get_Sequence(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    assert( curr_task != NULL);
    return (Quark_Sequence *)curr_task->sequence;
}

/* **************************************************************************** */
/**
 * For the current thread, in the current task being executed, return
 * the task label.  This is the value that was optionally provided
 * when the task was Task_Inserted.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @return Pointer to null-terminated label string
 * @return NULL if there is no label
 * @ingroup QUARK
 */
char *QUARK_Get_Task_Label(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    assert( curr_task != NULL);
    return (char *)curr_task->task_label;
}


/* **************************************************************************** */
/**
 * When a task is completed, queue it for further handling by another
 * process.
 */
static void worker_remove_completed_task_enqueue_for_later_processing(Quark *quark, Task *task, int worker_rank)
{
    int threads_remaining_for_this_task = -1;
    pthread_mutex_lock_wrap( &task->task_mutex );
    threads_remaining_for_this_task = --task->task_thread_count;
    pthread_mutex_unlock_wrap( &task->task_mutex );
    if ( threads_remaining_for_this_task == 0 ) {
        completed_tasks_node_t *node = malloc(sizeof(completed_tasks_node_t));
        node->task = task;
        node->workerid = worker_rank;
        pthread_mutex_lock_completed_tasks( &quark->completed_tasks_mutex );
        TAILQ_INSERT_TAIL( quark->completed_tasks, node, entries );
        pthread_mutex_unlock_completed_tasks( &quark->completed_tasks_mutex );
    }
}

/* **************************************************************************** */
/**
 * Handle the queue of completed tasks.
 */
static void process_completed_tasks(Quark *quark)
{
    completed_tasks_node_t *node = NULL;
    do {
        node = NULL;
        if ( pthread_mutex_trylock_asn( &quark->address_set_mutex ) == 0 ) {
            if ( pthread_mutex_trylock_completed_tasks( &quark->completed_tasks_mutex ) == 0 ) {
                node = TAILQ_FIRST(quark->completed_tasks);
                if ( node!= NULL ) TAILQ_REMOVE( quark->completed_tasks, node, entries );
                pthread_mutex_unlock_completed_tasks( &quark->completed_tasks_mutex );
            }
            if ( node != NULL ) {
                remove_completed_task_and_check_for_ready( quark, node->task, node->workerid );
                free( node );
            }
            pthread_mutex_unlock_asn( &quark->address_set_mutex );
        }
    } while ( node != NULL );
}

/* **************************************************************************** */
/**
 * Handle a single completed task, finding its children and putting
 * the children that are ready to go (all dependencies satisfied) into
 * worker ready queues.
 */
static void remove_completed_task_and_check_for_ready(Quark *quark, Task *task, int worker_rank)
{
    if ( quark->dot_dag_enable ) {
        pthread_mutex_lock_wrap( &quark->dot_dag_mutex );
        if (task->tasklevel < 1) task->tasklevel=1;
        fprintf(dot_dag_file, "t%lld [fillcolor=\"%s\",label=\"%s\",style=filled];\n", task->taskid, task->task_color, task->task_label);
        /* Track the width of each task level */
        quark->tasklevel_width[task->tasklevel]++;
        /* fprintf(dot_dag_file, "// critical-path depth %ld \n", task->tasklevel ); */
        fprintf(dot_dag_file, "{rank=same;%lld;t%lld};\n", task->tasklevel, task->taskid );
        pthread_mutex_unlock_wrap( &quark->dot_dag_mutex );
    }

    /* For each dependency in the task that was completed */
    icl_list_t *dep_node;
    for (dep_node = icl_list_first(task->dependency_list);
         dep_node != NULL &&  dep_node->data!=NULL;
         dep_node = icl_list_next(task->dependency_list, dep_node)) {
        Dependency  *dep = (Dependency *)dep_node->data;
        Address_Set_Node *address_set_node = dep->address_set_node_ptr;

        /* Mark the address/data as having been written by worker_rank  */
        if ( dep->direction==OUTPUT || dep->direction==INOUT )
            address_set_node->last_thread = worker_rank;
        if ( quark->dot_dag_enable ) {
            if ( dep->direction==OUTPUT || dep->direction==INOUT ) {
                /* Track last writer and level, needed when this structure becomes empty */
                address_set_node->last_writer_taskid = task->taskid;
                address_set_node->last_writer_tasklevel = task->tasklevel;
            }
            address_set_node->last_reader_or_writer_taskid = task->taskid;
            address_set_node->last_reader_or_writer_tasklevel = task->tasklevel;
        }
        /* Check the address set node to avoid WAR dependencies; if
         * just completed a write, and at least one more write
         * (sum>=2) is pending */
        if ( (quark->war_dependencies_enable) &&
             (dep->direction==OUTPUT || dep->direction==INOUT) &&
             ((address_set_node->num_waiting_output + address_set_node->num_waiting_inout) >= 2) ) {
            quark_avoid_war_dependencies( quark, address_set_node, task );
        }
        /* Remove competed dependency from address_set_node waiting_deps list */
        icl_list_delete( address_set_node->waiting_deps, dep->address_set_waiting_deps_node_ptr, NULL );
        /* Check initial INPUT next_deps attached to address_set_node */
        address_set_node_initial_input_check_and_launch( quark, address_set_node, dep, worker_rank );
       /* Handle any initial GATHERV dependencies */
       address_set_node_initial_gatherv_check_and_launch(quark, address_set_node, dep, worker_rank);
        /* Prepend any initial accumulater dependency that is ready to go */
        address_set_node_accumulator_find_prepend( quark, address_set_node );
        /* Check initial OUTPUT/INOUT deps waiting on address_set_node  */
        address_set_node_initial_output_check_and_launch( quark, address_set_node, dep, worker_rank );
        /* Keep track of the waiting dependency counts for this address */
        if (dep->direction == INPUT) address_set_node->num_waiting_input--;
        else if (dep->direction == OUTPUT) address_set_node->num_waiting_output--;
        else if (dep->direction == INOUT) address_set_node->num_waiting_inout--;

        /* If this address_set_node has no more waiting_deps, remove it */
        if ( icl_list_first(address_set_node->waiting_deps) == NULL )
            address_set_node_delete( quark, address_set_node );
    }

    task_delete(quark, task);
    quark->num_queued_tasks--;
}

/* **************************************************************************** */
/**
 * Set various task level flags.  This flag data structure is then
 * provided when the task is created/inserted.  Each flag can take a
 * value which is either an integer or a pointer.
 *
 *          Select from one of the flags:
 *          TASK_PRIORITY : an integer (0-MAX_INT)
 *          TASK_LOCK_TO_THREAD : an integer for the thread number
 *          TASK_LABEL : a string pointer (NULL terminated) for the label
 *          TASK_COLOR :  a string pointer (NULL terminated) for the color.
 *          TASK_SEQUENCE : takes pointer to a Quark_Sequence structure
 *
 * @param[in,out] flags
 *          Pointer to a Quark_Task_Flags structure
 * @param[in] flag
 *          One of the flags ( TASK_PRIORITY, TASK_LOCK_TO_THREAD, TASK_LABEL, TASK_COLOR, TASK_SEQUENCE )
 * @param[in] val
 *          A integer or a pointer value for the flag ( uses the intptr_t )
 * @return Pointer to the updated Quark_Task_Flags structure
 * @ingroup QUARK
 */
Quark_Task_Flags *QUARK_Task_Flag_Set( Quark_Task_Flags *task_flags, int flag, intptr_t val )
{
    switch (flag)  {
    case TASK_PRIORITY:
        task_flags->task_priority = (int)val;
        break;
    case TASK_LOCK_TO_THREAD:
        task_flags->task_lock_to_thread = (int)val;
        break;
    case TASK_LABEL:
        task_flags->task_label = (char *)val;
        break;
    case TASK_COLOR:
        task_flags->task_color = (char *)val;
        break;
    case TASK_SEQUENCE:
        task_flags->task_sequence = (Quark_Sequence *)val;
        break;
    case TASK_THREAD_COUNT:
        task_flags->task_thread_count = (int)val;
        break;
    }
    return task_flags;
}


