/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
#ifndef MAGMA_AMC_UTILS_H
#define MAGMA_AMC_UTILS_H

#ifndef min
#define min(a,b)   ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b)   ((a > b) ? a : b)
#endif


/* Abort the execution and print a message*/
int magma_amc_abort(const char *pattern, ...);

/* Return the recommanded percentage of the matrix for the CPU part*/
double magma_amc_recommanded_dcpu(int nbThreads, double cpu_peak, int nbGPUs, double gpu_peak);

/*split N in two part according to a ratio, and return the size of the first part*/
int NSplit(int N, double ratio);



/* Number of blocks for thread tid after a distribution over P threads*/
int numBlock2p(int tid, int NBblock, int P);

/* local position of thread tid in a block column distributed over P threads */
int indexBlock2p(int tid, int NBblock, int P);

#endif
