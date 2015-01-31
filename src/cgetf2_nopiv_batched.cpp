/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

   @author Azzam Haidar
   @author Adrien Remy

   @generated from zgetf2_nopiv_batched.cpp normal z -> c, Fri Jan 30 19:00:19 2015
*/

#include "common_magma.h"
#include "batched_kernel_param.h"
#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

/////////////////////////////////////////////////////////////////////////////////////////////////////
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    CGETF2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0 and N <= 1024.
            On CUDA architecture 1.x cards, N <= 512.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    ===================================================================== */

extern "C" magma_int_t
magma_cgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dW0_displ,
    magmaFloatComplex **dW1_displ,
    magmaFloatComplex **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount,
    cublasHandle_t myhandle, magma_queue_t queue)

{

    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
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

    magmaFloatComplex neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex one  = MAGMA_C_ONE;
    magma_int_t nb = 32;//BATF2_NB;

    
    magma_int_t min_mn = min(m, n);
    magma_int_t gbj, panelj, step, ib;

    for( panelj=0; panelj < min_mn; panelj+=nb) 
    {
        ib = min(nb, min_mn-panelj);

        for(step=0; step < ib; step++){
            gbj = panelj+step;
#if 0
            size_t required_shmem_size = ((m-panelj)*ib)*sizeof(magmaFloatComplex);
            if( required_shmem_size >  (MAX_SHARED_ALLOWED*1024))
#else
            if( (m-panelj) > 0)
#endif
            {
                // Compute elements J+1:M of J-th column.
                if (gbj < m) {
                    arginfo = magma_cscal_cgeru_batched(m-gbj, ib-step, gbj, dA_array, lda, info_array, gbstep, batchCount, queue);
                    if(arginfo != 0 ) return arginfo;
                }
            }
            else{
                // TODO
            }
        }


        if( (n-panelj-ib) > 0){
            // continue the update of the selected ib row column panelj+ib:n(TRSM)
            magma_cgetf2trsm_batched(ib, n-panelj-ib, dA_array, panelj, lda, batchCount, queue);
            // do the blocked DGER = DGEMM for the remaining panelj+ib:n columns
            magma_cdisplace_pointers(dW0_displ, dA_array, lda, ib+panelj, panelj, batchCount, queue);
            magma_cdisplace_pointers(dW1_displ, dA_array, lda, panelj, ib+panelj, batchCount, queue);            
            magma_cdisplace_pointers(dW2_displ, dA_array, lda, ib+panelj, ib+panelj, batchCount, queue);


#if 1
            magmablas_cgemm_batched( MagmaNoTrans, MagmaNoTrans, m-(panelj+ib), n-(panelj+ib), ib, 
                                      neg_one, dW0_displ, lda, 
                                      dW1_displ, lda, 
                                      one,  dW2_displ, lda, 
                                      batchCount, queue);
#else
            cublasCgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_N, m-(panelj+ib), n-(panelj+ib), ib,
                                     &neg_one, (const magmaFloatComplex**) dW0_displ, lda,
                                               (const magmaFloatComplex**) dW1_displ, lda,
                                     &one,  dW2_displ, lda, batchCount );
#endif
        }
    }

    //free(cpuAarray);

    return 0;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////

