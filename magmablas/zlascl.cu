/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define zlascl_bs 64


__global__ void
l_zlascl (int m, int n, double mul, magmaDoubleComplex* A, int lda){
    int ind =  blockIdx.x * zlascl_bs + threadIdx.x ;

    int break_d = (ind < n)? ind: n-1;

    A += ind;
    if (ind < m)
       for(int j=0; j<=break_d; j++ )
           A[j*lda] *= mul;
}

__global__ void
u_zlascl (int m, int n, double mul, magmaDoubleComplex* A, int lda){
    int ind =  blockIdx.x * zlascl_bs + threadIdx.x ;

    A += ind;
    if (ind < m)
      for(int j=n-1; j>= ind; j--)
         A[j*lda] *= mul;
}


extern "C" void
magmablas_zlascl(char type, magma_int_t kl, magma_int_t ku, 
                 double cfrom, double cto,
                 magma_int_t m, magma_int_t n, 
                 magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info )
{
    int blocks;
    if (m % zlascl_bs==0)
        blocks = m/ zlascl_bs;
    else
        blocks = m/ zlascl_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(zlascl_bs, 1, 1);

    /* To do : implment the accuracy procedure */
    double mul = cto / cfrom;

    if (type == 'L' || type =='l')  
       l_zlascl <<< grid, threads, 0, magma_stream >>> (m, n, mul, A, lda);
    else if (type == 'U' || type =='u')
       u_zlascl <<< grid, threads, 0, magma_stream >>> (m, n, mul, A, lda);  
    else {
       printf("Only type L and U are available in zlascl. Exit.\n");
       exit(1);
    }
}


