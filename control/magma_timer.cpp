/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016
*/

#include "magma_internal.h"

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


// =============================================================================
// Emulate gettimeofday on Windows.

#if defined( _WIN32 ) || defined( _WIN64 )
#ifndef _TIMEZONE_DEFINED
#define _TIMEZONE_DEFINED
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};
#endif

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


/***************************************************************************//**
    @return Current wall-clock time in seconds.
            Resolution is from gettimeofday.

    @ingroup magma_wtime
*******************************************************************************/
extern "C"
double magma_wtime( void )
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}


/***************************************************************************//**
    Calls magma_queue_sync() to synchronize, then returns current time.
    
    @param[in] queue    Queue to synchronize.

    @return Current wall-clock time in seconds.
            Resolution is from gettimeofday.

    @ingroup magma_wtime
*******************************************************************************/
extern "C"
double magma_sync_wtime( magma_queue_t queue )
{
    magma_queue_sync( queue );
    return magma_wtime();
}


#define magmaf_wtime FORTRAN_NAME( magmaf_wtime, MAGMAF_WTIME )

/***************************************************************************//**
    Version of magma_wtime() that is callable from Fortran.

    @param[out]
    time    On output, set to current wall-clock time in seconds.
            Resolution is from gettimeofday.

    @ingroup magma_wtime
*******************************************************************************/
extern "C"
void magmaf_wtime(double *time)
{
    *time = magma_wtime();
}
