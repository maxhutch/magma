/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:45 2013
       @author Ichitaro Yamazaki
*/
#include "common_magma.h"
#define PRECISION_s
#include "commonblas.h"

//
//    m, n - dimensions in the source (input) matrix.
//             This routine copies the ha matrix from the CPU
//             to dat on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_ssetmatrix_transpose_mgpu(
                  magma_int_t ngpus, magma_queue_t stream[][2],
                  const float *ha,  magma_int_t lda, 
                  float       *dat[], magma_int_t ldda, 
                  float       *db[],  magma_int_t lddb,
                  magma_int_t m, magma_int_t n, magma_int_t nb)
{
#define   A(j)    (ha       + (j)*lda)
#define  dB(d, j) (db[(d)]  + (j)*nb*lddb)
#define dAT(d, j) (dat[(d)] + (j)*nb)
    magma_int_t nstreams = 2, d, j, j_local, id, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || ngpus*ldda < n || lddb < m){
        printf( "Wrong arguments in magmablas_ssetmatrix_transpose_mgpu (%d<%d), (%d*%d<%d), or (%d<%d).\n",
                (int) lda, (int) m, (int) ngpus, (int) ldda, (int) n, (int) lddb, (int) m );
        return;
    }
    
    /* Move data from CPU to GPU by block columns and transpose it */
    for(j=0; j<n; j+=nb){
       d       = (j/nb)%ngpus;
       j_local = (j/nb)/ngpus;
       id      = j_local%nstreams;
       magma_setdevice(d);

       ib = min(n-j, nb);
       magma_ssetmatrix_async( m, ib,
                               A(j),      lda,
                               dB(d, id), lddb, 
                               stream[d][id] );

       magmablasSetKernelStream(stream[d][id]);
       magmablas_stranspose2(dAT(d, j_local), ldda, 
                             dB(d, id),       lddb, 
                             m, ib);
    }
}

