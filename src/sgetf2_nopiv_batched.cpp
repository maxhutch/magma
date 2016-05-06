/*
   -- MAGMA (version 2.0.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date May 2016

   @author Azzam Haidar
   @author Adrien Remy

   @generated from src/zgetf2_nopiv_batched.cpp normal z -> s, Mon May  2 23:30:25 2016
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#define A(i, j)  (A + (i) + (j)*ldda)   // A(i, j) means at i row, j column

///////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    SGETF2 computes an LU factorization of a general M-by-N matrix A without pivoting

    The factorization has the form
        A = L * U
    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param
    dW0_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dW1_displ (workspace) Array of pointers, dimension (batchCount).

    @param
    dW2_displ (workspace) Array of pointers, dimension (batchCount).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sgesv_aux
    ********************************************************************/

extern "C" magma_int_t
magma_sgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t ldda,
    float **dW0_displ,
    float **dW1_displ,
    float **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ldda < max(1,m)) {
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    float c_neg_one = MAGMA_S_NEG_ONE;
    float c_one     = MAGMA_S_ONE;
    magma_int_t nb = BATF2_NB;

    
    magma_int_t min_mn = min(m, n);
    magma_int_t gbj, panelj, step, ib;

    for( panelj=0; panelj < min_mn; panelj += nb) 
    {
        ib = min(nb, min_mn-panelj);

        for (step=0; step < ib; step++) {
            gbj = panelj+step;
#if 0
            size_t required_shmem_size = ((m-panelj)*ib)*sizeof(float);
            if ( required_shmem_size >  (MAX_SHARED_ALLOWED*1024))
#else
            if ( (m-panelj) > 0)
#endif
            {
                // Compute elements J+1:M of J-th column.
                if (gbj < m) {
                    arginfo = magma_sscal_sger_batched( m-gbj, ib-step, gbj, dA_array, ldda, info_array, gbstep, batchCount, queue );
                    if (arginfo != 0 ) return arginfo;
                }
            }
            else {
                // TODO
            }
        }


        if ( (n-panelj-ib) > 0) {
            // continue the update of the selected ib row column panelj+ib:n(TRSM)
            magma_sgetf2trsm_batched(ib, n-panelj-ib, dA_array, panelj, ldda, batchCount, queue);
            // do the blocked DGER = DGEMM for the remaining panelj+ib:n columns
            magma_sdisplace_pointers(dW0_displ, dA_array, ldda, ib+panelj, panelj, batchCount, queue);
            magma_sdisplace_pointers(dW1_displ, dA_array, ldda, panelj, ib+panelj, batchCount, queue);            
            magma_sdisplace_pointers(dW2_displ, dA_array, ldda, ib+panelj, ib+panelj, batchCount, queue);

            magma_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m-(panelj+ib), n-(panelj+ib), ib, 
                                 c_neg_one, dW0_displ, ldda, 
                                            dW1_displ, ldda, 
                                 c_one,     dW2_displ, ldda, 
                                 batchCount, queue );
        }
    }

    //magma_free_cpu(cpuAarray);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
