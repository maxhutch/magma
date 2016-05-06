/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt
       @author Eduardo Ponce

       @generated from sparse-iter/src/magma_zqr_wrapper.cpp normal z -> s, Mon May  2 23:31:02 2016
*/

#include "magmasparse_internal.h"


/**
    Purpose
    -------

    This is a wrapper to call MAGMA QR on the data structure of sparse matrices.
    Output matrices Q and R reside on the same memory location as matrix A.
    On exit, Q is a M-by-N matrix in lda-by-N space.
    On exit, R is a min(M,N)-by-N upper trapezoidal matrix. 


    Arguments
    ---------

    @param[in]
    m           magma_int_t
                dimension m
                
    @param[in]
    n           magma_int_t
                dimension n
                
    @param[in]
    A           magma_s_matrix
                input matrix A
                
    @param[in]
    lda         magma_int_t
                leading dimension matrix A
                
    @param[in,out]
    Q           magma_s_matrix*
                input matrix Q
                
    @param[in,out]
    R           magma_s_matrix*
                input matrix R

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sposv
    ********************************************************************/


extern "C" magma_int_t
magma_sqr(
    magma_int_t m, magma_int_t n,
    magma_s_matrix A, 
    magma_int_t lda, 
    magma_s_matrix *Q, 
    magma_s_matrix *R,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    // local constants
    const float c_zero = MAGMA_S_ZERO;

    // local variables
    magma_int_t inc = 1;
    magma_int_t k = min(m,n);
    magma_int_t ldt;
    magma_int_t nb;
    float *tau = NULL;
    float *dT = NULL;
    float *dA = NULL;
    magma_s_matrix dR1 = {Magma_CSR};

    // allocate CPU resources
    CHECK( magma_smalloc_pinned( &tau, k ) );

    // query number of blocks required for QR factorization
    nb = magma_get_sgeqrf_nb( m, n );
    ldt = (2 * k + magma_roundup(n, 32)) * nb;
    CHECK( magma_smalloc( &dT, ldt ) );

    // get copy of matrix array
    if ( A.memory_location == Magma_DEV ) {
        dA = A.dval;
    } else {
        CHECK( magma_smalloc( &dA, lda * n ) );
        magma_ssetvector( lda * n, A.val, inc, dA, inc, queue );
    }

    // QR factorization
    magma_sgeqrf_gpu( m, n, dA, lda, tau, dT, &info );  

    // construct R matrix
    if ( R != NULL ) {
        if ( A.memory_location == Magma_DEV ) {
            CHECK( magma_svinit( R, Magma_DEV, lda, n, c_zero, queue ) );
            magmablas_slacpy( MagmaUpper, k, n, dA, lda, R->dval, lda, queue );
        } else {
            CHECK( magma_svinit( &dR1, Magma_DEV, lda, n, c_zero, queue ) );
            magmablas_slacpy( MagmaUpper, k, n, dA, lda, dR1.dval, lda, queue );
            CHECK( magma_svinit( R, Magma_CPU, lda, n, c_zero, queue ) );
            magma_sgetvector( lda * n, dR1.dval, inc, R->val, inc, queue );
        }
    }

    // construct Q matrix
    if ( Q != NULL ) {
        magma_sorgqr_gpu( m, n, k, dA, lda, tau, dT, nb, &info ); 

        if ( A.memory_location == Magma_DEV ) {
            CHECK( magma_svinit( Q, Magma_DEV, lda, n, c_zero, queue ) );
            magma_scopyvector( lda * n, dA, inc, Q->dval, inc, queue );
        } else {
            CHECK( magma_svinit( Q, Magma_CPU, lda, n, c_zero, queue ) );
            magma_sgetvector( lda * n, dA, inc, Q->val, inc, queue );
        }
    }

cleanup:
    if( info != 0 ){
        magma_smfree( Q, queue );
        magma_smfree( R, queue );
        magma_smfree( &dR1, queue );
    }

    // free resources
    magma_free_pinned( tau );
    magma_free( dT );
    if ( A.memory_location == Magma_CPU ) {
        magma_free( dA );
    }

    return info;
}
