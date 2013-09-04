#ifndef PTHREAD_BARRIER_H
#define PTHREAD_BARRIER_H

#ifdef __APPLE__

#include <pthread.h>

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

#endif        // __APPLE__

#endif        //  #ifndef PTHREAD_BARRIER_H
