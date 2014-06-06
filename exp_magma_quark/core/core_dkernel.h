#ifndef CORE_DKERNEL_H
#define CORE_DKERNEL_H

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
#endif

#endif
