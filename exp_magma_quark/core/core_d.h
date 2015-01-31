/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
#ifndef CORE_D_H
#define CORE_D_H

void CORE_zgetrf_reclap_init();

int CORE_zgetrf_reclap(int M, int N,
                       double *A, int LDA,
                       int *IPIV, int *info); //core_zgetrf_reclap.cpp

#ifndef USE_CUDBG
//#include <mkl.h>
#define dgetrf dgetrf_
#define dtrsm dtrsm_
#define dlaswp dlaswp_
#define dgemm dgemm_
#define dlacpy dlacpy_
#endif


void core_dtslu_alloc(int nbThreads, int m, int nb);
void core_dtslu_init(int nbThreads);
void core_dtslu_free();

/*Do the preprocessing step of TSLU on a matrix A to factorize*/
void core_dtslu(int M, int nb, double *A, int LDA, int *IPIV, int *iinfo, int num_threads, int locNum);
#endif

