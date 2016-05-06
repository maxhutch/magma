/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zgerbt.cu normal z -> s, Mon May  2 23:30:30 2016


       @author Adrien REMY
*/
#include "magma_internal.h"
#include "sgerbt.h"

#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    SPRBT_MVT compute B = UTB to randomize B
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in]
    du     REAL array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in,out]
    db     REAL array, dimension (n)
            The n vector db computed by SGESV_NOPIV_GPU
            On exit db = du*db
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    ********************************************************************/
extern "C" void
magmablas_sprbt_mtv_q(
    magma_int_t n, 
    float *du, float *db,
    magma_queue_t queue)
{
    /*

     */
    magma_int_t threads = block_length;
    magma_int_t grid = magma_ceildiv( n, 4*block_length );

    magmablas_sapply_transpose_vector_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, du, n, db, 0);
    magmablas_sapply_transpose_vector_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, du, n+n/2, db, n/2);

    threads = block_length;
    grid = magma_ceildiv( n, 2*block_length );
    magmablas_sapply_transpose_vector_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n, du, 0, db, 0);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    SPRBT_MV compute B = VB to obtain the non randomized solution
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.
    
    @param[in,out]
    db      REAL array, dimension (n)
            The n vector db computed by SGESV_NOPIV_GPU
            On exit db = dv*db
    
    @param[in]
    dv      REAL array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    ********************************************************************/
extern "C" void
magmablas_sprbt_mv_q(
    magma_int_t n, 
    float *dv, float *db,
    magma_queue_t queue)
{
    magma_int_t threads = block_length;
    magma_int_t grid = magma_ceildiv( n, 2*block_length );

    magmablas_sapply_vector_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n, dv, 0, db, 0);

    threads = block_length;
    grid = magma_ceildiv( n, 4*block_length );

    magmablas_sapply_vector_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dv, n, db, 0);
    magmablas_sapply_vector_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dv, n+n/2, db, n/2);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    SPRBT randomize a square general matrix using partial randomized transformation
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns and rows of the matrix dA.  n >= 0.
    
    @param[in,out]
    dA      REAL array, dimension (n,ldda)
            The n-by-n matrix dA
            On exit dA = duT*dA*d_V
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDA >= max(1,n).
    
    @param[in]
    du      REAL array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix U
    
    @param[in]
    dv      REAL array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    ********************************************************************/
extern "C" void 
magmablas_sprbt_q(
    magma_int_t n, 
    float *dA, magma_int_t ldda, 
    float *du, float *dv,
    magma_queue_t queue)
{
    du += ldda;
    dv += ldda;

    dim3 threads(block_height, block_width);
    dim3 grid( magma_ceildiv( n, 4*block_height ), 
               magma_ceildiv( n, 4*block_width  ));

    magmablas_selementary_multiplication_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA,            0, ldda, du,   0, dv,   0);
    magmablas_selementary_multiplication_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA,     ldda*n/2, ldda, du,   0, dv, n/2);
    magmablas_selementary_multiplication_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA,          n/2, ldda, du, n/2, dv,   0);
    magmablas_selementary_multiplication_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA, ldda*n/2+n/2, ldda, du, n/2, dv, n/2);

    dim3 threads2(block_height, block_width);
    dim3 grid2( magma_ceildiv( n, 2*block_height ), 
                magma_ceildiv( n, 2*block_width  ));
    magmablas_selementary_multiplication_kernel<<< grid2, threads2, 0, queue->cuda_stream() >>>(n, dA, 0, ldda, du, -ldda, dv, -ldda);
}
