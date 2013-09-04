/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @author Azzam Haidar
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
        printf("==========================================================================================\n");
        printf("  WARNING a parallel section (D&C) that use OPENMP could not perform in parallel because  \n");
        printf("  it need to be compiled with multi thread library and add -fopenmp(gcc)/-openmp(icc)     \n");
        printf("  to both compilation and linkage flags.                                                  \n");
        printf("  If you wish to remove the printout of this WARNING, please compile with -DNO_WARNING    \n");
        printf("==========================================================================================\n");
        #endif
#else
   #ifndef NO_WARNING	
   printf("==========================================================================================\n");
   printf("  WARNING you are calling a parallel section without linking with a multithread library   \n");
   printf("  please compile with multi thread library and add -fopenmp(gcc)/-openmp(icc) to both     \n");
   printf("  compilation and linkage flags.                                                          \n");
   printf("  If you wish to remove the printout of this WARNING, please compile with -DNO_WARNING    \n");
   printf("==========================================================================================\n");
   #endif
#endif
}
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
magma_int_t magma_get_numthreads()
{
    /* determine the number of threads */
    magma_int_t threads = 0;
    char *env;
    // First check MKL_NUM_THREADS if MKL is used
#if defined(MAGMA_WITH_MKL)
    env = getenv("MKL_NUM_THREADS");
    if (env != NULL)
        threads = atoi(env);
#endif
    // Second check OMP_NUM_THREADS
    if (threads < 1){
        env = getenv("OMP_NUM_THREADS");
        if (env != NULL)
            threads = atoi(env);
    }
    // Third use the number of CPUs
    if (threads < 1)
        threads = sysconf(_SC_NPROCESSORS_ONLN);
    // Fourth use one thread
    if (threads < 1)
        threads = 1;

    return threads;
}
/////////////////////////////////////////////////////////////










