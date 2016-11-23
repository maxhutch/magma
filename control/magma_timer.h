/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016
       
       @author Mark Gates
*/

#ifndef MAGMA_TIMER_H
#define MAGMA_TIMER_H

#include <stdio.h>

#include "magma_v2.h"

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

/***************************************************************************//**
    @param[out]
    t       On output, set to current time.
    
    If ENABLE_TIMER is not defined, does nothing.
    
    @ingroup magma_timer
*******************************************************************************/
static inline void timer_start( magma_timer_t &t )
{
    #if defined(ENABLE_TIMER)
    t = magma_wtime();
    #endif
}


/***************************************************************************//**
    @param[out]
    t       On output, set to current time.
    
    @param[in]
    queue  Queue to sync with, before getting time.
    
    If ENABLE_TIMER is not defined, does nothing.
    
    @ingroup magma_timer
*******************************************************************************/
static inline void timer_sync_start( magma_timer_t &t, magma_queue_t queue )
{
    #if defined(ENABLE_TIMER)
    magma_queue_sync( queue );
    t = magma_wtime();
    #endif
}


/***************************************************************************//**
    @param[in,out]
    t       On input, time when timer_start() was called.
            On output, set to (current time - start time).

    @return t, to sum up times:
    
        magma_timer_t time, time_sum=0;
        for( ... ) {
            timer_start( time );
            ...do timed operations...
            time_sum += timer_stop( time );
        
            ...do other operations...
        }
    
    If ENABLE_TIMER is not defined, returns 0.
    
    @ingroup magma_timer
*******************************************************************************/
static inline magma_timer_t timer_stop( magma_timer_t &t )
{
    #if defined(ENABLE_TIMER)
    t = magma_wtime() - t;
    return t;
    #else
    return 0;
    #endif
}


/***************************************************************************//**
    @param[in,out]
    t       On input, time when timer_start() was called.
            On output, set to (current time - start time).
    
    @param[in]
    queue  Queue to sync with, before getting time.

    @return t, to sum up times:
    
        magma_timer_t time, time_sum=0;
        for( ... ) {
            timer_start( time );
            ...do timed operations...
            time_sum += timer_stop( time );
        
            ...do other operations...
        }
    
    If ENABLE_TIMER is not defined, returns 0.
    
    @ingroup magma_timer
*******************************************************************************/
static inline magma_timer_t timer_sync_stop( magma_timer_t &t, magma_queue_t queue )
{
    #if defined(ENABLE_TIMER)
    magma_queue_sync( queue );
    t = magma_wtime() - t;
    return t;
    #else
    return 0;
    #endif
}


/***************************************************************************//**
    @param[out]
    flops   On output, set to current flop counter.
    
    Requires global gPAPI_flops_set to be setup by testing/magma_util.cpp
    Note that newer CPUs may not support flop counts; see
    https://icl.cs.utk.edu/projects/papi/wiki/PAPITopics:SandyFlops
    
    If ENABLE_TIMER and HAVE_PAPI are not both defined, does nothing.
    
    @ingroup magma_timer
*******************************************************************************/
static inline void flops_start( magma_flops_t &flops )
{
    #if defined(ENABLE_TIMER) && defined(HAVE_PAPI)
    PAPI_read( gPAPI_flops_set, &flops );
    #endif
}


/***************************************************************************//**
    @param[out]
    flops   On input, flop counter when flops_start() was called.
            On output, set to (current flop counter - start flop counter).
    
    @return flops, so you can sum up; see timer_stop().
    
    If ENABLE_TIMER and HAVE_PAPI are not both defined, returns 0.
    
    @ingroup magma_timer
*******************************************************************************/
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


/***************************************************************************//**
    If ENABLE_TIMER is defined, same as printf;
    else does nothing (returns 0).
    
    @ingroup magma_timer
*******************************************************************************/
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


/***************************************************************************//**
    If ENABLE_TIMER is defined, same as fprintf;
    else does nothing (returns 0).
    
    @ingroup magma_timer
*******************************************************************************/
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


/***************************************************************************//**
    If ENABLE_TIMER is defined, same as snprintf;
    else does nothing (returns 0).
    
    @ingroup magma_timer
*******************************************************************************/
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
