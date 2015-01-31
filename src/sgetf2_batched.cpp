/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from zgetf2_batched.cpp normal z -> s, Fri Jan 30 19:00:19 2015
       @author Azzam Haidar
       @author Tingxing Dong
*/

#include "common_magma.h"
#include "batched_kernel_param.h"

#define PRECISION_s

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

/////////////////////////////////////////////////////////////////////////////////////////////////////
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    SGETF2 computes an LU factorization of a general m-by-n matrix A
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

    A       (input/output) REAL array, dimension (LDA,N)
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
magma_sgetf2_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **dW0_displ,
    float **dW1_displ,
    float **dW2_displ,
    magma_int_t **ipiv_array,
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

    float neg_one = MAGMA_S_NEG_ONE;
    float one  = MAGMA_S_ONE;
    magma_int_t nb = BATF2_NB;

    

    //float **cpuAarray = (float**) malloc(batchCount*sizeof(float*));
    //magma_getvector( batchCount, sizeof(float*), dA_array, 1, cpuAarray, 1);


    magma_int_t min_mn = min(m, n);
    magma_int_t gbj, panelj, step, ib;

    for( panelj=0; panelj < min_mn; panelj+=nb) 
    {
        ib = min(nb, min_mn-panelj);

        for(step=0; step < ib; step++){
            gbj = panelj+step;
            //size_t required_shmem_size = zamax*(sizeof(float)+sizeof(int)) + (m-panelj+2)*sizeof(float);
            //if( (m-panelj) > 0)
            if( (m-panelj) > MAX_NTHREADS)
            //if( required_shmem_size >  (MAX_SHARED_ALLOWED*1024))
            {
                //printf("running non shared version\n");
                // find the max of the column gbj
                arginfo = magma_isamax_batched(m-gbj, dA_array, 1, gbj, lda, ipiv_array, info_array, gbstep, batchCount, queue);
                if(arginfo != 0 ) return arginfo;
                // Apply the interchange to columns 1:N. swap the whole row
                arginfo = magma_sswap_batched(n, dA_array, lda, gbj, ipiv_array, batchCount, queue);
                if(arginfo != 0 ) return arginfo;
                // Compute elements J+1:M of J-th column.
                if (gbj < m) {
                    arginfo = magma_sscal_sger_batched(m-gbj, ib-step, gbj, dA_array, lda, info_array, gbstep, batchCount, queue);
                    if(arginfo != 0 ) return arginfo;
                }
            }
            else{
                //printf("running --- shared version\n");                
                arginfo = magma_scomputecolumn_batched(m-panelj, panelj, step, dA_array, lda, ipiv_array, info_array, gbstep, batchCount, queue);
                if(arginfo != 0 ) return arginfo;
                // Apply the interchange to columns 1:N. swap the whole row
                arginfo = magma_sswap_batched(n, dA_array, lda, gbj, ipiv_array, batchCount, queue);
                if(arginfo != 0 ) return arginfo;
            }
        }


        if( (n-panelj-ib) > 0){
            // continue the update of the selected ib row column panelj+ib:n(TRSM)
            magma_sgetf2trsm_batched(ib, n-panelj-ib, dA_array, panelj, lda, batchCount, queue);
            // do the blocked DGER = DGEMM for the remaining panelj+ib:n columns
            magma_sdisplace_pointers(dW0_displ, dA_array, lda, ib+panelj, panelj, batchCount, queue);
            magma_sdisplace_pointers(dW1_displ, dA_array, lda, panelj, ib+panelj, batchCount, queue);            
            magma_sdisplace_pointers(dW2_displ, dA_array, lda, ib+panelj, ib+panelj, batchCount, queue);


#if 1
            magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m-(panelj+ib), n-(panelj+ib), ib, 
                                      neg_one, dW0_displ, lda, 
                                      dW1_displ, lda, 
                                      one,  dW2_displ, lda, 
                                      batchCount, queue);
#else
            cublasSgemmBatched(myhandle, CUBLAS_OP_N, CUBLAS_OP_N, m-(panelj+ib), n-(panelj+ib), ib,
                                     &neg_one, (const float**) dW0_displ, lda,
                                               (const float**) dW1_displ, lda,
                                     &one,  dW2_displ, lda, batchCount );
#endif
        }
    }

    //free(cpuAarray);

    return 0;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////

