/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
*/

#include "common_magma.h"

#if defined( _WIN32 ) || defined( _WIN64 )
#  include <time.h>
#  include <sys/timeb.h>
#  if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#  else
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#  endif
#else
#  include <sys/time.h>
#endif

#if defined(ADD_)
#  define magma_wtime_f        magma_wtime_f_
#elif defined(NOCHANGE)
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Emulate gettimeofday on Windows.
*/ 
#if defined( _WIN32 ) || defined( _WIN64 )
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};

extern "C"
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME         ft;
    unsigned __int64 tmpres = 0;
    static int       tzflag = 0;

    if (NULL != tv) {
        GetSystemTimeAsFileTime(&ft);
        tmpres |=  ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |=  ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres /= 10;  /*convert into microseconds*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;

        tv->tv_sec  = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }
    if (NULL != tz) {
        if (!tzflag) {
            _tzset();
            tzflag = 1;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime     = _daylight;
    }
    return 0;
}
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Get current time.
*/ 
extern "C"
magma_timestr_t get_current_time(void)
{
    struct timeval  time_val;
    magma_timestr_t time;
    
    cudaDeviceSynchronize();
    gettimeofday(&time_val, NULL);
    
    time.sec  = time_val.tv_sec;
    time.usec = time_val.tv_usec;
    return (time);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Returns elapsed time between start and end in milliseconds.
*/ 
extern "C"
double GetTimerValue(magma_timestr_t start, magma_timestr_t end)
{
    int sec, usec;
    
    sec  = end.sec  - start.sec;
    usec = end.usec - start.usec;
    
    return (1000.*(double)(sec) + (double)(usec) * 0.001);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return time in seconds since arbitrary time in the past.
      Use for elapsed wall clock time computation.
*/
extern "C"
double magma_wtime( void )
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}

// synchronize before getting time, e.g., to time asynchronous cublas calls
extern "C"
double magma_sync_wtime( magma_queue_t queue )
{
    magma_queue_sync( queue );
    return magma_wtime();
}

// version callable from Fortran stores seconds in time.
extern "C"
void magma_wtime_f(double *time)
{
    *time = magma_wtime();
}
