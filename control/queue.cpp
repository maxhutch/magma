#include <stdio.h>
#include <string.h>

#include "queue.hpp"

// If err, prints error and throws exception.
void check( int err )
{
    if ( err != 0 ) {
        fprintf( stderr, "Error: %s (%d)\n", strerror(err), err );
        throw std::exception();
    }
}


// ---------------------------------------------
// thread's main routine
// executes tasks from queue (given as arg), until a NULL task is returned.
void* magma_thread_main( void* arg )
{
    magma_queue* queue = (magma_queue*) arg;
    magma_task* task;
    
    while( true ) {
        task = queue->pop_task();
        if ( task == NULL ) {
            break;
        }
        
        task->run();
        queue->finish_task( task );
        delete task;
        task = NULL;
    }
    
    return NULL;  // implicitly does pthread_exit
}


// ---------------------------------------------
// Creates queue with NO threads.
magma_queue::magma_queue():
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


// Calls quit, then deallocates data.
magma_queue::~magma_queue()
{
    quit();
    check( pthread_mutex_destroy( &mutex ));
    check( pthread_cond_destroy( &cond ));
    check( pthread_cond_destroy( &cond_ntask ));
}


// Allocates data and creates nthread threads.
void magma_queue::launch( magma_int_t in_nthread )
{
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


// Add task to queue.
// Increments number of outstanding tasks.
// Signals one thread that is waiting.
void magma_queue::push_task( magma_task* task )
{
    check( pthread_mutex_lock( &mutex ));
    if ( quit_flag ) {
        fprintf( stderr, "Error: push_task() called after quit()\n" );
        throw std::exception();
    }
    q.push( task );
    ntask += 1;
    //printf( "push; ntask %d\n", ntask );
    check( pthread_cond_signal( &cond ));
    check( pthread_mutex_unlock( &mutex ));
}


// Returns next task, or NULL if queue is empty AND quit is set to true,
// blocking if necesary.
// This does NOT decrement number of outstanding tasks;
// thread should call finish_task() when work is quit.
magma_task* magma_queue::pop_task()
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


// Marks task as finished, decrementing number of outstanding tasks.
// Signals one thread that is waiting.
// Currently, task is not used.
void magma_queue::finish_task( magma_task* task )
{
    check( pthread_mutex_lock( &mutex ));
    ntask -= 1;
    //printf( "fini; ntask %d\n", ntask );
    check( pthread_cond_signal( &cond_ntask ));
    check( pthread_mutex_unlock( &mutex ));
}


// Block until all outstanding tasks have been finished.
void magma_queue::sync()
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


// Sets quit_flag, so pop_task will return NULL once queue is empty,
// telling thread to exit.
// Signals all threads that are waiting.
// Joins all threads.
// Safe to call quit multiple times -- the first time all the threads are
// joined, subsequent times it does nothing.
// (Destructor also calls quit, but you may prefer to call it explicitly.)
void magma_queue::quit()
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
        for( magma_int_t i=0; i < nthread; ++i ) {
            check( pthread_join( threads[i], NULL ));
            //printf( "joined %d (%lx)\n", i, (long) threads[i] );
        }
        if ( threads != NULL ) {
            delete[] threads;
            threads = NULL;
        }
    }
}


// Mostly for debugging, returns thread index in range 0, ..., nthread-1.
magma_int_t magma_queue::get_thread_index( pthread_t thread ) const
{
    for( magma_int_t i=0; i < nthread; ++i ) {
        if ( pthread_equal( thread, threads[i] )) {
            return i;
        }
    }
    return -1;
}
