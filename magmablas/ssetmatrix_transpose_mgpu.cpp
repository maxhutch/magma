/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zsetmatrix_transpose_mgpu.cpp normal z -> s, Mon May  2 23:30:38 2016
       @author Ichitaro Yamazaki
*/
#include "magma_internal.h"

//
//    m, n - dimensions in the source (input) matrix.
//             This routine copies the hA matrix from the CPU
//             to dAT on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddw*nb pointed to by dwork (lddw > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_ssetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    const float *hA,       magma_int_t lda, 
    magmaFloat_ptr    dAT[],    magma_int_t ldda, 
    magmaFloat_ptr    dwork[],  magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb)
{
#define hA(j)       (hA         + (j)*lda)
#define dwork(d, j) (dwork[(d)] + (j)*nb*lddw)
#define dAT(d, j)   (dAT[(d)]   + (j)*nb)

    magma_int_t nqueues = 2, d, j, j_local, id, ib;
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;
    
    if (lda < m || ngpu*ldda < n || lddw < m) {
        fprintf( stderr, "%s: wrong arguments (%d<%d), (%d*%d<%d), or (%d<%d).\n",
                 __func__, (int) lda, (int) m, (int) ngpu, (int) ldda, (int) n, (int) lddw, (int) m );
        return;
    }
    
    /* Move data from CPU to GPU by block columns and transpose it */
    for (j=0; j < n; j += nb) {
        d       = (j/nb)%ngpu;
        j_local = (j/nb)/ngpu;
        id      = j_local%nqueues;
        magma_setdevice(d);
        
        ib = min(n-j, nb);
        magma_ssetmatrix_async( m, ib,
                                hA(j),        lda,
                                dwork(d, id), lddw, 
                                queues[d][id] );
        
        magmablas_stranspose_q( m, ib, dwork(d,id), lddw, dAT(d,j_local), ldda, queues[d][id] );
    }
}
