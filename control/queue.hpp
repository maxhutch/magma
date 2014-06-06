#ifndef MAGMA_THREAD_HPP
#define MAGMA_THREAD_HPP

#include <pthread.h>
#include <queue>

#include "magma.h"


// ---------------------------------------------
class magma_task
{
public:
    magma_task() {}
    virtual ~magma_task() {}
    
    virtual void run() = 0;  // pure virtual function to execute task
};


// ---------------------------------------------
// Multi-threaded work pool.
class magma_queue
{
public:
    magma_queue();
    ~magma_queue();
    
    void launch( magma_int_t in_nthread );
    void push_task( magma_task* task );
    magma_task* pop_task();
    void finish_task( magma_task* task );
    void sync();
    void quit();
    
    magma_int_t get_thread_index( pthread_t thread ) const;
    
protected:
    std::queue< magma_task* > q;  // queue of tasks
    bool            quit_flag;    // quit() sets this to true; after this, pop returns NULL
    magma_int_t     ntask;        // number of unfinished tasks (in queue or currently executing)
    pthread_mutex_t mutex;        // mutex lock for queue, quit, ntask
    pthread_cond_t  cond;         // condition variable for changes to queue and quit (push, pop, quit)
    pthread_cond_t  cond_ntask;   // condition variable for changes to ntask (sync)
    pthread_t*      threads;      // array of threads
    magma_int_t     nthread;      // number of threads
};

#endif        //  #ifndef MAGMA_THREAD_HPP
