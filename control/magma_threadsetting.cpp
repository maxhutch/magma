/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Azzam Haidar
       @author Simplice Donfack
*/
#include "common_magma.h"

/***************************************************************************//**
 * switch lapack thread_num initialization
 **/
#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#endif

/////////////////////////////////////////////////////////////
void magma_setlapack_numthreads(magma_int_t num_threads)
{
#if defined(_OPENMP)
    omp_set_num_threads( num_threads );
    #if defined(MAGMA_WITH_MKL)
    mkl_set_num_threads( num_threads );
    #endif
#elif defined(MAGMA_WITH_MKL)
    mkl_set_num_threads( num_threads );
    #ifndef NO_WARNING
    printf("==============================================================================\n"
           "WARNING: a parallel section of MAGMA could not be run in parallel because\n"
           "OpenMP was not enabled; add -fopenmp (for gcc) or -openmp (for icc) to\n"
           "both compilation and linkage flags (OPTS and LDOPTS in make.inc).\n"
           "To disable this warning, compile MAGMA with -DNO_WARNING.\n"
           "==============================================================================\n");
    #endif
#else
    #ifndef NO_WARNING
    printf("==============================================================================\n"
           "WARNING: a parallel section of MAGMA could not be run in parallel because\n"
           "OpenMP was not enabled; add -fopenmp (for gcc) or -openmp (for icc) to\n"
           "both compilation and linkage flags (OPTS and LDOPTS in make.inc).\n"
           "Also, if using multi-threaded MKL, please compile MAGMA with -DMAGMA_WITH_MKL.\n"
           "To disable this warning, compile MAGMA with -DNO_WARNING.\n"
           "==============================================================================\n");
    #endif
#endif
}


/////////////////////////////////////////////////////////////
magma_int_t magma_getlapack_numthreads()
{
    magma_int_t num_threads = -1;
#if defined(_OPENMP)
    num_threads = omp_get_num_threads();
    #if defined(MAGMA_WITH_MKL)
    num_threads = mkl_get_max_threads();
    #endif
#elif defined(MAGMA_WITH_MKL)
    num_threads = mkl_get_max_threads();
    #ifndef NO_WARNING
    printf("==============================================================================\n"
           "WARNING: a parallel section of MAGMA could not be run in parallel because\n"
           "OpenMP was not enabled; add -fopenmp (for gcc) or -openmp (for icc) to\n"
           "both compilation and linkage flags (OPTS and LDOPTS in make.inc).\n"
           "To disable this warning, compile MAGMA with -DNO_WARNING.\n"
           "==============================================================================\n");
    #endif
#else
   #ifndef NO_WARNING
    printf("==============================================================================\n"
           "WARNING: a parallel section of MAGMA could not be run in parallel because\n"
           "OpenMP was not enabled; add -fopenmp (for gcc) or -openmp (for icc) to\n"
           "both compilation and linkage flags (OPTS and LDOPTS in make.inc).\n"
           "Also, if using multi-threaded MKL, please compile MAGMA with -DMAGMA_WITH_MKL.\n"
           "To disable this warning, compile MAGMA with -DNO_WARNING.\n"
           "==============================================================================\n");
   #endif
#endif
    if(num_threads==-1)
        num_threads = magma_get_numthreads();
    return num_threads;
}


/////////////////////////////////////////////////////////////
magma_int_t magma_get_numthreads()
{
    /* determine the number of threads */
    magma_int_t threads = 0;
    char *myenv;

    // First check OMP_NUM_THREADS then MKL then the system CPUs
#if defined(_OPENMP)
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
#elif defined(MAGMA_WITH_MKL)
    threads = mkl_get_max_threads();
#else
    myenv = getenv("MAGMA_NUM_THREADS");
    if (myenv != NULL)
    {
        threads = atoi(myenv);
    }
    else{
        #ifdef _MSC_VER  // Windows
        SYSTEM_INFO sysinfo;
        GetSystemInfo( &sysinfo );
        threads = sysinfo.dwNumberOfProcessors;
        #else
        threads = sysconf(_SC_NPROCESSORS_ONLN);
        #endif
    }
#endif

    // Fourth use one thread
    if (threads < 1)
        threads = 1;

    return threads;
}
/////////////////////////////////////////////////////////////
