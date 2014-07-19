/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zgetmatrix_transpose.cu normal z -> s, Fri Jul 18 17:34:12 2014

*/
#include "common_magma.h"
#define PRECISION_s
#include "commonblas.h"


//
//      m, n - dimensions in the output (ha) matrix.
//             This routine copies the dat matrix from the GPU
//             to ha on the CPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_sgetmatrix_transpose( magma_int_t m, magma_int_t n,
                                const float *dat, magma_int_t ldda,
                                float       *ha,  magma_int_t lda,
                                float       *dB,  magma_int_t lddb, magma_int_t nb )
{
    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || ldda < n || lddb < m){
        printf("Wrong arguments in sgetmatrix_transpose.\n");
        return;
    }

    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    for(i=0; i < n; i += nb) {
       /* Move data from GPU to CPU using 2 buffers; 1st transpose the data on the GPU */
       ib = min(n-i, nb);

       magmablas_stranspose_stream( ib, m, dat+i, ldda, dB+(j%2)*nb*lddb, lddb, stream[j%2] );
       magma_sgetmatrix_async( m, ib,
                               dB + (j%2) * nb * lddb, lddb,
                               ha+i*lda,               lda, stream[j%2] );
       j++;
    }

    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
}
