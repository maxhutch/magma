/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
*/

#include "thread_queue.hpp"

// If err, prints error and throws exception.
static void check( int err )
{
    if ( err != 0 ) {
        fprintf( stderr, "Error: %s (%d)\n", strerror(err), err );
        throw std::exception();
    }
}


/**
    @class magma_thread_queue
    
    Purpose
    -------
    Implements a thread pool with a multi-producer, multi-consumer queue.
    
    Typical use:
    A main thread creates the queue and tells it to launch worker threads. Then
    the main thread inserts (pushes) tasks into the queue. Threads will execute
    the tasks. No dependencies are tracked. The main thread can sync the queue,
    waiting for all current tasks to finish, and then insert more tasks into the
    queue. When finished, the main thread calls quit or simply destructs the
    queue, which will exit all worker threads.
    
    Tasks are sub-classes of magma_task. They must implement the run() function.
    
    Example
    -------
    @code
    class task1: public magma_task {
    public:
        task1( int arg ):
            m_arg( arg ) {}
        
        virtual void run() { do_task1( m_arg ); }
    private:
        int m_arg;
    };
    
    class task2: public magma_task {
    public:
        task2( int arg1, int arg2 ):
            m_arg1( arg1 ), m_arg2( arg2 ) {}
        
        virtual void run() { do_task2( m_arg1, m_arg2 ); }
    private:
        int m_arg1, m_arg2;
    };
    
    void master( int n ) {
        magma_thread_queue queue;
        queue.launch( 12 );  // 12 worker threads
        for( int i=0; i < n; ++i ) {
            queue.push_task( new task1( i ));
        }
        queue.sync();  // wait for all task1 to finish before doing task2.
        for( int i=0; i < n; ++i ) {
            for( int j=0; j < i; ++j ) {
                queue.push_task( new task2( i, j ));
            }
        }
        queue.quit();  // [optional] explicitly exit worker threads
    }
    @endcode
    
    This is similar to python's queue class, but also implements worker threads
    and adds quit mechanism. sync is like python's join, but threads do not
    exit, so join would be a misleading name.
*/


// ---------------------------------------------
/// Thread's main routine, executed by pthread_create.
/// Executes tasks from queue (given as arg), until a NULL task is returned.
/// Deletes each task when it is done.
/// @param[in,out] arg    magma_thread_queue to get tasks from.
extern "C"
void* magma_thread_main( void* arg )
{
    magma_thread_queue* queue = (magma_thread_queue*) arg;
    magma_task* task;
    
    while( true ) {
        task = queue->pop_task();
        if ( task == NULL ) {
            break;
        }
        
        task->run();
        queue->task_done();
        delete task;
        task = NULL;
    }
    
    return NULL;  // implicitly does pthread_exit
}


// ---------------------------------------------
/// Creates queue with NO threads. Use \ref launch to create threads.
magma_thread_queue::magma_thread_queue():
    q        (),
    quit_flag( false ),
    ntask    ( 0     ),
    threads  ( NULL  ),
    nthread  ( 0     )
{
    check( pthread_mutex_init( &mutex,      NULL ));
    check( pthread_cond_init(  &cond,       NULL ));
    check( pthread_cond_init(  &cond_ntask, NULL ));
}


/// Calls \ref quit, then deallocates data.
magma_thread_queue::~magma_thread_queue()
{
    quit();
    check( pthread_mutex_destroy( &mutex ));
    check( pthread_cond_destroy( &cond ));
    check( pthread_cond_destroy( &cond_ntask ));
}


/// Creates threads.
/// @param[in] in_nthread    Number of threads to launch.
void magma_thread_queue::launch( magma_int_t in_nthread )
{
    assert( threads == NULL );  // else launch was called previously
    nthread = in_nthread;
    if ( nthread < 1 ) {
        nthread = 1;
    }
    threads = new pthread_t[ nthread ];
    for( magma_int_t i=0; i < nthread; ++i ) {
        check( pthread_create( &threads[i], NULL, magma_thread_main, this ));
        //printf( "launch %d (%lx)\n", i, (long) threads[i] );
    }
}


/// Add task to queue. Task must be allocated with C++ new.
/// Increments number of outstanding tasks.
/// Signals threads that are waiting in pop_task.
/// @param[in] task    Task to queue.
void magma_thread_queue::push_task( magma_task* task )
{
    check( pthread_mutex_lock( &mutex ));
    if ( quit_flag ) {
        fprintf( stderr, "Error: push_task() called after quit()\n" );
        throw std::exception();
    }
    q.push( task );
    ntask += 1;
    //printf( "push; ntask %d\n", ntask );
    check( pthread_cond_broadcast( &cond ));
    check( pthread_mutex_unlock( &mutex ));
}


/// Get next task from queue.
/// @return next task, blocking until a task is inserted if necesary.
/// @return NULL if queue is empty *and* \ref quit has been called.
///
/// This does *not* decrement number of outstanding tasks;
/// thread should call \ref task_done when task is completed.
magma_task* magma_thread_queue::pop_task()
{
    magma_task* task = NULL;
    check( pthread_mutex_lock( &mutex ));
    while( q.empty() && ! quit_flag ) {
        check( pthread_cond_wait( &cond, &mutex ));
    }
    // q has item or quit is set (or both)
    if ( ! q.empty()) {
        task = q.front();
        q.pop();
    }
    //printf( "pop;  ntask %d\n", ntask );
    check( pthread_mutex_unlock( &mutex ));
    return task;
}


/// Marks task as finished, decrementing number of outstanding tasks.
/// Signals threads that are waiting in \ref sync.
void magma_thread_queue::task_done()
{
    check( pthread_mutex_lock( &mutex ));
    ntask -= 1;
    //printf( "fini; ntask %d\n", ntask );
    check( pthread_cond_broadcast( &cond_ntask ));
    check( pthread_mutex_unlock( &mutex ));
}


/// Block until all outstanding tasks have been finished.
/// Threads continue to be alive; more tasks can be pushed after sync.
void magma_thread_queue::sync()
{
    check( pthread_mutex_lock( &mutex ));
    //printf( "sync; ntask %d [start]\n", ntask );
    while( ntask > 0 ) {
        check( pthread_cond_wait( &cond_ntask, &mutex ));
        //printf( "sync; ntask %d\n", ntask );
    }
    //printf( "sync; ntask %d [done]\n", ntask );
    check( pthread_mutex_unlock( &mutex ));
}


/// Sets quit_flag, so \ref pop_task will return NULL once queue is empty,
/// telling threads to exit.
/// Signals all threads that are waiting in pop_task.
/// Waits for all threads to exit (i.e., joins them).
/// It is safe to call quit multiple times -- the first time all the threads are
/// joined; subsequent times it does nothing.
/// (Destructor also calls quit, but you may prefer to call it explicitly.)
void magma_thread_queue::quit()
{
    // first, set quit_flag and signal waiting threads
    bool join = true;
    check( pthread_mutex_lock( &mutex ));
    //printf( "quit %d\n", quit_flag );
    if ( quit_flag ) {
        join = false;  // quit previously called; don't join again.
    }
    else {
        quit_flag = true;
        check( pthread_cond_broadcast( &cond ));
    }
    check( pthread_mutex_unlock( &mutex ));
    
    // next, join all threads
    if ( join ) {
        assert( threads != NULL );
        for( magma_int_t i=0; i < nthread; ++i ) {
            check( pthread_join( threads[i], NULL ));
            //printf( "joined %d (%lx)\n", i, (long) threads[i] );
        }
        delete[] threads;
        threads = NULL;
    }
}


/// Mostly for debugging, returns thread index in range 0, ..., nthread-1.
magma_int_t magma_thread_queue::get_thread_index( pthread_t thread ) const
{
    for( magma_int_t i=0; i < nthread; ++i ) {
        if ( pthread_equal( thread, threads[i] )) {
            return i;
        }
    }
    return -1;
}
