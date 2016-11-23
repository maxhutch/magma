/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/zgemv_vbatched.cpp, normal z -> d, Sun Nov 20 20:20:32 2016
       
       @author Ahmad Abdelfattah
*/
#include "magma_internal.h"
#include "commonblas_d.h"

#define PRECISION_d

/******************************************************************************/
extern "C" void
magmablas_dgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    info =  magma_gemv_vbatched_checker( trans, m, n, ldda, incx, incy, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmablas_dgemv_vbatched_max_nocheck( 
            trans, 
            m, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_m, max_n, queue);
}


/******************************************************************************/
extern "C" void
magmablas_dgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue)
{
    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

    magmablas_dgemv_vbatched_max_nocheck( 
            trans, 
            m, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_m, max_n, queue);
}


/***************************************************************************//**
    Purpose
    -------
    DGEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER M specifies the number of rows of each matrix A.
            The last element of the array is used internally by the routine. 

    @param[in]
    n       Array of integers, dimension (batchCount + 1).
            On entry, each INTEGER N specifies the number of columns of each matrix A
            The last element of the array is used internally by the routine. 
 
    @param[in]
    alpha   DOUBLE PRECISION
            On entry, ALPHA specifies the scalar alpha.


    @param[in]
    dA_array     Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array A of DIMENSION ( LDDA, N ) on the GPU
   
    @param[in]
    ldda    Array of integers, dimension (batchCount + 1).
            Each INTEGER LDDA specifies the leading dimension of each matrix A.

    @param[in]
    dx_array     Array of pointers, dimension (batchCount).
            Each is a DOUBLE PRECISION array of dimension
            N if trans == MagmaNoTrans
            M if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Array of integers, dimension (batchCount + 1).
            Each integer specifies the increment for the elements of each vector X.
            INCX must not be zero.
            The last element of the array is used internally by the routine. 
  
    @param[in]
    beta    DOUBLE PRECISION
            On entry, ALPHA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy_array     Array of pointers, dimension (batchCount).
            Each is a DOUBLE PRECISION array of dimension
            M if trans == MagmaNoTrans
            N if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Array of integers, dimension (batchCount + 1).
            Each integer specifies the increment for the elements of each vector Y.
            INCY must not be zero.
            The last element of the array is used internally by the routine. 

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv_batched
*******************************************************************************/
extern "C" void
magmablas_dgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    info =  magma_gemv_vbatched_checker( trans, m, n, ldda, incx, incy, batchCount, queue );
        
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // compute the max. dimensions
    magma_imax_size_2(m, n, batchCount, queue);
    magma_int_t max_m, max_n; 
    magma_igetvector_async(1, &m[batchCount], 1, &max_m, 1, queue);
    magma_igetvector_async(1, &n[batchCount], 1, &max_n, 1, queue);
    magma_queue_sync( queue );

    magmablas_dgemv_vbatched_max_nocheck( 
            trans, 
            m, n, 
            alpha, dA_array, ldda, 
                   dx_array, incx, 
            beta,  dy_array, incy, 
            batchCount, max_m, max_n, queue);
}
