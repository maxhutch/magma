/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014

       @generated from zlascl.cu normal z -> s, Fri Apr 25 15:05:21 2014

*/
#include "common_magma.h"

#define slascl_bs 64


__global__ void
l_slascl (int m, int n, float mul, float* A, int lda){
    int ind =  blockIdx.x * slascl_bs + threadIdx.x ;

    int break_d = (ind < n)? ind: n-1;

    A += ind;
    if (ind < m)
       for(int j=0; j<=break_d; j++ )
           A[j*lda] *= mul;
}

__global__ void
u_slascl (int m, int n, float mul, float* A, int lda){
    int ind =  blockIdx.x * slascl_bs + threadIdx.x ;

    A += ind;
    if (ind < m)
      for(int j=n-1; j>= ind; j--)
         A[j*lda] *= mul;
}


extern "C" void
magmablas_slascl(magma_type_t type, magma_int_t kl, magma_int_t ku, 
                 float cfrom, float cto,
                 magma_int_t m, magma_int_t n, 
                 float *A, magma_int_t lda, magma_int_t *info )
{
    int blocks;
    if (m % slascl_bs==0)
        blocks = m/ slascl_bs;
    else
        blocks = m/ slascl_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(slascl_bs, 1, 1);

    /* To do : implment the accuracy procedure */
    float mul = cto / cfrom;

    if (type == MagmaLower)  
       l_slascl <<< grid, threads, 0, magma_stream >>> (m, n, mul, A, lda);
    else if (type == MagmaUpper)
       u_slascl <<< grid, threads, 0, magma_stream >>> (m, n, mul, A, lda);  
    else {
       printf("Only type L and U are available in slascl. Exit.\n");
       exit(1);
    }
}
