/**
 *
 * @file quarkos.c
 *
 *  This file handles the mapping from pthreads calls to windows threads
 *  QUARK is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.3.1
 * @author Piotr Luszczek
 * @author Mathieu Faverge
 * @date January 2015
 *
 *  Note : this file is a copy of a PLASMA file for use of QUARK alone
 *
 **/

#if defined(linux) || defined(__linux) || defined(__linux__)
#define QUARK_OS_LINUX 1
#define _GNU_SOURCE
#include <unistd.h>
#include <sched.h>
#elif defined( _WIN32 ) || defined( _WIN64 )
#define QUARK_OS_WINDOWS 1
#include <Windows.h>
#elif (defined __APPLE__) || (defined macintosh) || (defined __MACOSX__)
#define QUARK_OS_MACOS 1
#include <sys/param.h>
#include <sys/sysctl.h>
#include <mach/mach_init.h>
#include <mach/thread_policy.h>
kern_return_t thread_policy_set(thread_act_t thread, thread_policy_flavor_t flavor,
                                thread_policy_t policy_info, mach_msg_type_number_t count);
#elif (defined _AIX)
#define QUARK_OS_AIX 1
#else
#error "Cannot find the runing system or system not supported. Please define try to QUARK_OS_[LINUX|MACOS|AIX|WINDOWS]"
#endif

#if defined(QUARK_HWLOC) && (defined QUARK_AFFINITY_DISABLE)
#undef QUARK_HWLOC
#endif

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
//#include "common.h"
#if defined( _WIN32 ) || defined( _WIN64 )
#include "quarkwinthread.h"
#else
#include <pthread.h>
#endif

#include "quark.h"
#define QUARK_SUCCESS 0
#define QUARK_ERR -1
#define QUARK_ERR_UNEXPECTED -1

// maximum cores per context
#define CONTEXT_THREADS_MAX  256


#ifdef __cplusplus
extern "C" {
#endif

static pthread_mutex_t  mutextopo = PTHREAD_MUTEX_INITIALIZER;
static volatile int sys_corenbr = 1;
static volatile int topo_initialized = 0;

  /*
   * Topology functions
   */
#ifdef QUARK_HWLOC
#include "quarkos-hwloc.c"
#else

void quark_topology_init(){
    pthread_mutex_lock(&mutextopo);
    if ( !topo_initialized ) {
#if (defined QUARK_OS_LINUX) || (defined QUARK_OS_AIX)

        sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);

#elif (defined QUARK_OS_MACOS)

        int mib[4];
        int cpu;
        size_t len = sizeof(cpu);

        /* set the mib for hw.ncpu */
        mib[0] = CTL_HW;
        mib[1] = HW_AVAILCPU;

        /* get the number of CPUs from the system */
        sysctl(mib, 2, &cpu, &len, NULL, 0);
        if( cpu < 1 ) {
            mib[1] = HW_NCPU;
            sysctl( mib, 2, &cpu, &len, NULL, 0 );
        }
        if( cpu < 1 ) {
            cpu = 1;
        }
        sys_corenbr = cpu;
#elif (defined QUARK_OS_WINDOWS)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        sys_corenbr = sysinfo.dwNumberOfProcessors;
#endif
    }
    pthread_mutex_unlock(&mutextopo);
}

void quark_topology_finalize(){}

/**
 This routine will set affinity for the calling thread that has rank 'rank'.
 Ranks start with 0.

 If there are multiple instances of QUARK then affinity will be wrong: all ranks 0
 will be pinned to core 0.

 Also, affinity is not resotred when QUARK_Finalize() is called.
 */
int quark_setaffinity(int rank) {
#ifndef QUARK_AFFINITY_DISABLE
#if (defined QUARK_OS_LINUX)
    {
        cpu_set_t set;
        CPU_ZERO( &set );
        CPU_SET( rank, &set );

#if (defined HAVE_OLD_SCHED_SETAFFINITY)
        if( sched_setaffinity( 0, &set ) < 0 )
#else /* HAVE_OLD_SCHED_SETAFFINITY */
        if( sched_setaffinity( 0, sizeof(set), &set) < 0 )
#endif /* HAVE_OLD_SCHED_SETAFFINITY */
            {
                return QUARK_ERR_UNEXPECTED;
            }

        return QUARK_SUCCESS;
    }
#elif (defined QUARK_OS_MACOS)
    {
        thread_affinity_policy_data_t ap;
        int                           ret;

        ap.affinity_tag = 1; /* non-null affinity tag */
        ret = thread_policy_set( mach_thread_self(),
                                 THREAD_AFFINITY_POLICY,
                                 (integer_t*) &ap,
                                 THREAD_AFFINITY_POLICY_COUNT
            );
        if(ret != 0)
            return QUARK_ERR_UNEXPECTED;

        return QUARK_SUCCESS;
    }
#elif (defined QUARK_OS_WINDOWS)
    {
        DWORD mask = 1 << rank;

        if( SetThreadAffinityMask(GetCurrentThread(), mask) == 0)
            return QUARK_ERR_UNEXPECTED;

        return QUARK_SUCCESS;
    }
#elif (defined QUARK_OS_AIX)
    {
        tid_t self_ktid = thread_self ();
        bindprocessor(BINDTHREAD, self_ktid, rank);
        return QUARK_SUCCESS;
    }
#else
    return QUARK_ERR_NOT_SUPPORTED;
#endif
#endif /* QUARK_AFFINITY_DISABLE */
    return QUARK_ERR_NOT_SUPPORTED;
}
#endif /* QUARK_HWLOC */

/** ****************************************************************************
   A thread can unlock the CPU if it has nothing to do to let
   another thread of less priority running for example for I/O.
 */
int quark_yield() {
#if (defined QUARK_OS_LINUX) || (defined QUARK_OS_MACOS) || (defined QUARK_OS_AIX)
    return sched_yield();
#elif QUARK_OS_WINDOWS
    return SleepEx(0,0);
#else
    return QUARK_ERR_NOT_SUPPORTED;
#endif
}

#ifdef QUARK_OS_WINDOWS
#define QUARK_GETENV(var, str) {                    \
        int len = 512;                               \
        int ret;                                     \
        str = (char*)malloc(len * sizeof(char));     \
        ret = GetEnvironmentVariable(var, str, len); \
        if (ret == 0) {                              \
            free(str);                               \
            str = NULL;                              \
        }                                            \
    }

#define QUARK_CLEANENV(str) if (str != NULL) free(str);

#else /* Other OS systems */

#define QUARK_GETENV(var, str)  envstr = getenv(var);
#define QUARK_CLEANENV(str)

#endif

/** ****************************************************************************
 * Check for an integer in an environment variable, returning the
 * integer value or a provided default value
*/
int quark_get_numthreads()
{
    char    *envstr  = NULL;
    char    *endptr;
    long int thrdnbr = -1;
    extern int errno;

    /* Env variable does not exist, we search the system number of core */
    QUARK_GETENV("QUARK_NUM_THREADS", envstr);
    if ( envstr == NULL ) {
        thrdnbr = sys_corenbr;
    } else {
        /* Convert to long, checking for errors */
        thrdnbr = strtol(envstr, &endptr, 10);
        if ((errno == ERANGE) || ((thrdnbr==0) && (endptr==envstr))) {
            QUARK_CLEANENV(envstr);
            return -1;
        }
    }
    QUARK_CLEANENV(envstr);
    return (int)thrdnbr;
}

int *quark_get_affthreads(/* int *coresbind */) {
    char *envstr = NULL;
    int i;
    
    int *coresbind = (int *)malloc(CONTEXT_THREADS_MAX*sizeof(int));
    /* Env variable does not exist, we search the system number of core */
    QUARK_GETENV("QUARK_AFF_THREADS", envstr);
    if ( envstr == NULL) {
        for (i = 0; i < CONTEXT_THREADS_MAX; i++)
            coresbind[i] = i % sys_corenbr;
    }
    else {
        char *endptr;
        int   wrap = 0;
        int   nbr  = 0;
        long int val;

        /* We use the content of the QUARK_AFF_THREADS env. variable */
        for (i = 0; i < CONTEXT_THREADS_MAX; i++) {
            if (!wrap) {
                val = strtol(envstr, &endptr, 10);
                if (endptr != envstr) {
                    coresbind[i] = (int)val;
                    envstr = endptr;
                }
                else {
                    /* there must be at least one entry */
                    if (i < 1) {
                        //quark_error("quark_get_affthreads", "QUARK_AFF_THREADS should have at least one entry => everything will be bind on core 0");
                        fprintf(stderr, "quark_get_affthreads: QUARK_AFF_THREADS should have at least one entry => everything will be bind on core 0");
                        coresbind[i] = 0;
                        i++;
                    }

                    /* there is no more values in the string                                 */
                    /* the next threads are binded with a round robin policy over this array */
                    wrap = 1;
                    nbr = i;

                    coresbind[i] = coresbind[0];
                }
            }
            else {
                coresbind[i] = coresbind[i % nbr];
            }
        }
    }
    QUARK_CLEANENV(envstr);
    /* return QUARK_SUCCESS; */
    return coresbind;
}


/** ****************************************************************************
*/
int quark_getenv_int(char* name, int defval)
{
    char    *envstr  = NULL;
    char    *endptr;
    long int longval = -1;
    extern int errno;

    QUARK_GETENV(name, envstr);
    if ( envstr == NULL ) {
        longval = defval;
    } else {
        /* Convert to long, checking for errors */
        longval = strtol(envstr, &endptr, 10);
        if ((errno == ERANGE) || ((longval==0) && (endptr==envstr))) {
            longval = defval;
        }
    }
    QUARK_CLEANENV(envstr);
    return (int)longval;
}


#ifdef __cplusplus
}
#endif

