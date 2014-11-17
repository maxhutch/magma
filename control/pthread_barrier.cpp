// MacOS doesn't provide pthread_barrier
// magmawinthread.c doesn't provide pthread_barrier
#if (defined( _WIN32 ) || defined( _WIN64 ) || defined( __APPLE__ )) && ! defined( __MINGW32__ )

#include <errno.h>

#include "pthread_barrier.h"

// from http://stackoverflow.com/questions/3640853/performance-test-sem-t-v-s-dispatch-semaphore-t-and-pthread-once-t-v-s-dispat

// *sigh* OSX does not have pthread_barrier
int pthread_barrier_init( pthread_barrier_t *barrier,
                          const pthread_barrierattr_t *attr, unsigned int count )
{
    if ( count == 0 ) {
        errno = EINVAL;
        return -1;
    }
    if ( pthread_mutex_init( &barrier->mutex, 0 ) < 0 ) {
        return -1;
    }
    if ( pthread_cond_init( &barrier->cond, 0 ) < 0 ) {
        pthread_mutex_destroy( &barrier->mutex );
        return -1;
    }
    barrier->tripCount = count;
    barrier->count = 0;

    return 0;
}

int pthread_barrier_destroy( pthread_barrier_t *barrier )
{
    pthread_cond_destroy( &barrier->cond );
    pthread_mutex_destroy( &barrier->mutex );
    return 0;
}

int pthread_barrier_wait( pthread_barrier_t *barrier )
{
    pthread_mutex_lock( &barrier->mutex );
    ++(barrier->count);
    if ( barrier->count >= barrier->tripCount ) {
        barrier->count = 0;
        pthread_cond_broadcast( &barrier->cond );
        pthread_mutex_unlock( &barrier->mutex );
        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    else {
        pthread_cond_wait( &barrier->cond, &(barrier->mutex) );
        pthread_mutex_unlock( &barrier->mutex );
        return 0;
    }
}

#endif // (_WIN32 || _WIN64 || __APPLE__) && ! __MINGW32__
