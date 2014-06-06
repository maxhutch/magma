/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:45 2013

*/
#include "common_magma.h"


/*********************************************************/
/*
 *  Swap diagonal blocks of two matrices. 
 *  For more detail see the description below.
 */

__global__ void 
magmagpu_sswapdblk(int nb,
                   float *dA1, int ldda1, int inca1,
                   float *dA2, int ldda2, int inca2 )
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    dA1 += tx + bx * nb * (ldda1 + inca1);
    dA2 += tx + bx * nb * (ldda2 + inca2);

    float tmp;

    #pragma unroll
    for( int i = 0; i < nb; i++ ){
        tmp = dA1[i*ldda1];
        dA1[i*ldda1] = dA2[i*ldda2];
        dA2[i*ldda2] = tmp;
    }
}


extern "C" void 
magmablas_sswapdblk(magma_int_t n, magma_int_t nb,
                    float *dA1, magma_int_t ldda1, magma_int_t inca1,
                    float *dA2, magma_int_t ldda2, magma_int_t inca2 )
{
/* -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    This is an auxiliary MAGMA routine. It swaps diagonal blocks
    of size nb x nb between matrices dA1 and dA2 on the GPU.

    The number of blocks swapped is (n-1)/nb. For i = 1 .. (n-1)/nb matrices
    dA1 + i * nb * (ldda1 + inca1) and
    dA2 + i * nb * (ldda2 + inca2) are swapped.
*/

    magma_int_t blocksize = nb;
    dim3 blocks( (n-1) / blocksize, 1, 1);

    magmagpu_sswapdblk<<< blocks, blocksize, 0, magma_stream >>>( nb, 
                                                 dA1, ldda1, inca1,
                                                 dA2, ldda2, inca2 );
}

