/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar

*/
#include "magma_internal.h"

#if defined(linux) || defined(__linux) || defined(__linux__)
    #define MAGMA_OS_LINUX 1
    #ifndef _GNU_SOURCE
    #define _GNU_SOURCE
    #endif

    #include <unistd.h>
    #include <sched.h>

#elif defined(__FreeBSD__)
    #define MAGMA_OS_FREEBSD 1
    #include <unistd.h>
    #include <inttypes.h>
    #include <sys/param.h>
    #include <sys/cpuset.h>
    #include <sched.h>

#elif defined( _WIN32 ) || defined( _WIN64 )
    #define MAGMA_OS_WINDOWS 1
    #include <Windows.h>

#elif (defined __APPLE__) || (defined macintosh) || (defined __MACOSX__)
    #define MAGMA_OS_MACOS 1
    #include <sys/param.h>
    #include <sys/sysctl.h>
    #include <mach/mach_init.h>
    #include <mach/thread_policy.h>

#elif (defined _AIX)
    #define MAGMA_OS_AIX 1

#else
    #error "Cannot determine the OS, or OS not supported. Supported systems are: Linux, FreeBSD, Windows, MacOS X, AIX."
#endif

/** ****************************************************************************
   A thread can unlock the CPU if it has nothing to do to let
   another thread of less priority running for example for I/O.
 */
magma_int_t magma_yield()
{
#if (defined MAGMA_OS_LINUX) || (defined MAGMA_OS_FREEBSD) || (defined MAGMA_OS_MACOS) || (defined MAGMA_OS_AIX)
    return sched_yield();
#elif MAGMA_OS_WINDOWS
    return SleepEx(0,0);
#else
    return MAGMA_ERR_NOT_SUPPORTED;
#endif
}
