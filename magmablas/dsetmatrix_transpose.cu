/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from zsetmatrix_transpose.cu normal z -> d, Tue Sep  2 12:38:16 2014

*/
#include "common_magma.h"

#define PRECISION_d


//
//      m, n - dimensions in the source (input) matrix.
//             This routine copies the hA matrix from the CPU
//             to dAT on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_dsetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    const double  *hA, magma_int_t lda, 
    double       *dAT, magma_int_t ldda,
    double        *dB, magma_int_t lddb, magma_int_t nb,
    magma_queue_t queues[2] )
{
    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    // TODO standard check arguments
    if (lda < m || ldda < n || lddb < m){
        printf("Wrong arguments in %s.\n", __func__);
        return;
    }

    /* Move data from CPU to GPU in the first panel in the dB buffer */
    ib = min(n-i, nb);
    magma_dsetmatrix_async( m, ib,
                            hA + i*lda,             lda,
                            dB + (j%2) * nb * lddb, lddb, queues[j%2] );
    j++;

    for(i=nb; i < n; i += nb) {
       /* Move data from CPU to GPU in the second panel in the dB buffer */
       ib = min(n-i, nb);
       magma_dsetmatrix_async( m, ib,
                               hA+i*lda,               lda,
                               dB + (j%2) * nb * lddb, lddb, queues[j%2] );
       j++;
  
       /* Note that the previous panel (i.e., j%2) comes through the queue
          for the kernel so there is no need to synchronize.             */
       // TODO should this be ib not nb?
       magmablas_dtranspose_q( m, nb, dB+(j%2)*nb*lddb, lddb, dAT+i-nb, ldda, queues[j%2] );
    }

    /* Transpose the last part of the matrix.                            */
    j++;
    magmablas_dtranspose_q( m, ib, dB+(j%2)*nb*lddb, lddb, dAT+i-nb, ldda, queues[j%2] );
}


// @see magmablas_dsetmatrix_transpose_q
extern "C" void 
magmablas_dsetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const double  *hA, magma_int_t lda, 
    double       *dAT, magma_int_t ldda,
    double        *dB, magma_int_t lddb, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_dsetmatrix_transpose_q( m, n, hA, lda, dAT, ldda, dB, lddb, nb, queues );
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}
