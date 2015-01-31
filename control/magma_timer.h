/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Mark Gates
*/

#ifndef MAGMA_TIMER_H
#define MAGMA_TIMER_H

#include <stdio.h>

#include "magma.h"

typedef double    magma_timer_t;
typedef long long magma_flops_t;

#if defined(ENABLE_TIMER)
    #include <stdio.h>
    #include <stdarg.h>
    
    #if defined(HAVE_PAPI)
        #include <papi.h>
        extern int gPAPI_flops_set;  // defined in testing/magma_util.cpp
    #endif
#endif

// If we're not using GNU C, elide __attribute__
#ifndef __GNUC__
  #define  __attribute__(x)  /*NOTHING*/
#endif


// ------------------------------------------------------------
// Set timer to current time.
// If ENABLE_TIMER is not defined, does nothing.
static inline void timer_start( magma_timer_t &t )
{
    #if defined(ENABLE_TIMER)
    t = magma_wtime();
    #endif
}

// Set timer to difference between current time and when timer_start() was called.
// Returns timer, to sum up times:
//
// magma_timer_t time, time_sum=0;
// for( ... ) {
//     timer_start( time );
//     ...do timed operations...
//     time_sum += timer_stop( time );
//
//     ...do other operations...
// }
//
// If ENABLE_TIMER is not defined, returns 0.
static inline magma_timer_t timer_stop( magma_timer_t &t )
{
    #if defined(ENABLE_TIMER)
    t = magma_wtime() - t;
    return t;
    #else
    return 0;
    #endif
}


// ------------------------------------------------------------
// see gPAPI_flops_set in testing/magma_util.cpp

// Set flops counter to current flops.
// If ENABLE_TIMER and HAVE_PAPI are not both defined, does nothing.
static inline void flops_start( magma_flops_t &flops )
{
    #if defined(ENABLE_TIMER) && defined(HAVE_PAPI)
    PAPI_read( gPAPI_flops_set, &flops );
    #endif
}

// Set flops counter to difference between current flops and when flops_start() was called.
// Returns counter, so you can sum up; see timer_stop().
// If ENABLE_TIMER and HAVE_PAPI are not both defined, returns 0.
static inline magma_flops_t flops_stop( magma_flops_t &flops )
{
    #if defined(ENABLE_TIMER) && defined(HAVE_PAPI)
    magma_flops_t end;
    PAPI_read( gPAPI_flops_set, &end );
    flops = end - flops;
    return flops;
    #else
    return 0;
    #endif
}


// ------------------------------------------------------------
// If ENABLE_TIMER is defined, same as printf;
// else does nothing (returns 0).
static inline int timer_printf( const char* format, ... )
    __attribute__((format(printf,1,2)));

static inline int timer_printf( const char* format, ... )
{
    int len = 0;
    #if defined(ENABLE_TIMER)
    va_list ap;
    va_start( ap, format );
    len = vprintf( format, ap );
    va_end( ap );
    #endif
    return len;
}

// If ENABLE_TIMER is defined, same as fprintf;
// else does nothing (returns 0).
static inline int timer_fprintf( FILE* stream, const char* format, ... )
    __attribute__((format(printf,2,3)));

static inline int timer_fprintf( FILE* stream, const char* format, ... )
{
    int len = 0;
    #if defined(ENABLE_TIMER)
    va_list ap;
    va_start( ap, format );
    len = vfprintf( stream, format, ap );
    va_end( ap );
    #endif
    return len;
}

// If ENABLE_TIMER is defined, same as snprintf;
// else does nothing (returns 0).
static inline int timer_snprintf( char* str, size_t size, const char* format, ... )
    __attribute__((format(printf,3,4)));

static inline int timer_snprintf( char* str, size_t size, const char* format, ... )
{
    int len = 0;
    #if defined(ENABLE_TIMER)
    va_list ap;
    va_start( ap, format );
    len = vsnprintf( str, size, format, ap );
    va_end( ap );
    #endif
    return len;
}

#endif        //  #ifndef MAGMA_TIMER_H
