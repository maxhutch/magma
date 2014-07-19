/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zsetmatrix_transpose.cu normal z -> s, Fri Jul 18 17:34:13 2014

*/
#include "common_magma.h"
#define PRECISION_s
#include "commonblas.h"


//
//      m, n - dimensions in the source (input) matrix.
//             This routine copies the ha matrix from the CPU
//             to dat on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_ssetmatrix_transpose( magma_int_t m, magma_int_t n,
                                const float  *ha, magma_int_t lda, 
                                float       *dat, magma_int_t ldda,
                                float        *dB, magma_int_t lddb, magma_int_t nb )
{
    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || ldda < n || lddb < m){
        printf("Wrong arguments in %s.\n", __func__);
        return;
    }

    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
   
    /* Move data from CPU to GPU in the first panel in the dB buffer */
    ib = min(n-i, nb);
    magma_ssetmatrix_async( m, ib,
                            ha + i*lda,             lda,
                            dB + (j%2) * nb * lddb, lddb, stream[j%2] );
    j++;

    for(i=nb; i < n; i += nb) {
       /* Move data from CPU to GPU in the second panel in the dB buffer */
       ib = min(n-i, nb);
       magma_ssetmatrix_async( m, ib,
                               ha+i*lda,               lda,
                               dB + (j%2) * nb * lddb, lddb, stream[j%2] );
       j++;
  
       /* Note that the previous panel (i.e., j%2) comes through the stream
          for the kernel so there is no need to synchronize.             */
       // TODO should this be ib not nb?
       magmablas_stranspose_stream( m, nb, dB+(j%2)*nb*lddb, lddb, dat+i-nb, ldda, stream[j%2] );
    }

    /* Transpose the last part of the matrix.                            */
    j++;
    magmablas_stranspose_stream( m, ib, dB+(j%2)*nb*lddb, lddb, dat+i-nb, ldda, stream[j%2] );

    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
}
