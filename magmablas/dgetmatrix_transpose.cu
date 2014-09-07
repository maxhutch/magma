/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from zgetmatrix_transpose.cu normal z -> d, Tue Sep  2 12:38:16 2014

*/
#include "common_magma.h"

#define PRECISION_d


//
//      m, n - dimensions in the output (hA) matrix.
//             This routine copies the dAT matrix from the GPU
//             to hA on the CPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_dgetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    const double *dAT, magma_int_t ldda,
    double       *hA,  magma_int_t lda,
    double       *dB,  magma_int_t lddb, magma_int_t nb,
    magma_queue_t queues[2] )
{
    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    // TODO standard check arguments
    if (lda < m || ldda < n || lddb < m){
        printf("Wrong arguments in dgetmatrix_transpose.\n");
        return;
    }

    for(i=0; i < n; i += nb) {
       /* Move data from GPU to CPU using 2 buffers; 1st transpose the data on the GPU */
       ib = min(n-i, nb);

       magmablas_dtranspose_q( ib, m, dAT+i, ldda, dB+(j%2)*nb*lddb, lddb, queues[j%2] );
       magma_dgetmatrix_async( m, ib,
                               dB + (j%2) * nb * lddb, lddb,
                               hA+i*lda,               lda, queues[j%2] );
       j++;
    }
}


// @see magmablas_dgetmatrix_transpose_q
extern "C" void 
magmablas_dgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const double *dAT, magma_int_t ldda,
    double       *hA,  magma_int_t lda,
    double       *dB,  magma_int_t lddb, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_dgetmatrix_transpose_q( m, n, dAT, ldda, hA, lda, dB, lddb, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}
