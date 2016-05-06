/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

//
//      m, n - dimensions in the output (hA) matrix.
//             This routine copies the dAT matrix from the GPU
//             to hA on the CPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddwork*nb pointed to by dwork (lddwork > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_zgetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dAT, magma_int_t ldda,
    magmaDoubleComplex          *hA,  magma_int_t lda,
    magmaDoubleComplex_ptr       dwork,  magma_int_t lddwork, magma_int_t nb,
    magma_queue_t queues[2] )
{
#define    hA(i_, j_)    (hA + (i_) + (j_)*lda)
#define   dAT(i_, j_)   (dAT + (i_) + (j_)*ldda)
#define dwork(i_, j_) (dwork + (i_) + (j_)*lddwork)

    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    // TODO standard check arguments
    if (lda < m || ldda < n || lddwork < m) {
        fprintf( stderr, "%s: wrong arguments.\n", __func__ );
        return;
    }

    for (i=0; i < n; i += nb) {
        /* Move data from GPU to CPU using 2 buffers; 1st transpose the data on the GPU */
        ib = min(n-i, nb);
        
        magmablas_ztranspose_q( ib, m, dAT(i,0), ldda, dwork(0,(j%2)*nb), lddwork, queues[j%2] );
        magma_zgetmatrix_async( m, ib,
                                dwork(0,(j%2)*nb), lddwork,
                                hA(0,i), lda, queues[j%2] );
        j++;
    }
}
