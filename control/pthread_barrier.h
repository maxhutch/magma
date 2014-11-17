#ifndef PTHREAD_BARRIER_H
#define PTHREAD_BARRIER_H

// MacOS doesn't provide pthread_barrier
// magmawinthread.c doesn't provide pthread_barrier
#if (defined( _WIN32 ) || defined( _WIN64 ) || defined( __APPLE__ )) && ! defined( __MINGW32__ )

#if defined( _WIN32 ) || defined( _WIN64 )
    #include "magmawinthread.h"
#else
    #include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// from http://stackoverflow.com/questions/3640853/performance-test-sem-t-v-s-dispatch-semaphore-t-and-pthread-once-t-v-s-dispat

// *sigh* OSX does not have pthread_barrier
typedef int pthread_barrierattr_t;
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} pthread_barrier_t;

#define PTHREAD_BARRIER_SERIAL_THREAD 1

int pthread_barrier_init( pthread_barrier_t *barrier,
                          const pthread_barrierattr_t *attr, unsigned int count );

int pthread_barrier_destroy( pthread_barrier_t *barrier );

int pthread_barrier_wait( pthread_barrier_t *barrier );

#ifdef __cplusplus
}
#endif

#endif        // (_WIN32 || _WIN64 || __APPLE__) && ! __MINGW32__

#endif        //  #ifndef PTHREAD_BARRIER_H
