/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from zgerbt.cu normal z -> c, Fri Sep 11 18:29:20 2015


       @author Adrien REMY
*/
#include "common_magma.h"
#include "cgerbt.h"

#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    CPRBT_MVT compute B = UTB to randomize B
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in]
    du     COMPLEX array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in,out]
    db     COMPLEX array, dimension (n)
            The n vector db computed by CGESV_NOPIV_GPU
            On exit db = du*db
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    ********************************************************************/
extern "C" void
magmablas_cprbt_mtv_q(
    magma_int_t n, 
    magmaFloatComplex *du, magmaFloatComplex *db,
    magma_queue_t queue)
{
    /*

     */
    magma_int_t threads = block_length;
    magma_int_t grid = magma_ceildiv( n, 4*block_length );

    magmablas_capply_transpose_vector_kernel<<< grid, threads, 0, queue >>>(n/2, du, n, db, 0);
    magmablas_capply_transpose_vector_kernel<<< grid, threads, 0, queue >>>(n/2, du, n+n/2, db, n/2);

    threads = block_length;
    grid = magma_ceildiv( n, 2*block_length );
    magmablas_capply_transpose_vector_kernel<<< grid, threads, 0, queue >>>(n, du, 0, db, 0);
}

/**
    @see magmablas_cprbt_mtv_q
    ********************************************************************/
extern "C" void
magmablas_cprbt_mtv(
    magma_int_t n, 
    magmaFloatComplex *du, magmaFloatComplex *db)
{
    magmablas_cprbt_mtv_q(n, du, db, magma_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------
    CPRBT_MV compute B = VB to obtain the non randomized solution
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.
    
    @param[in,out]
    db      COMPLEX array, dimension (n)
            The n vector db computed by CGESV_NOPIV_GPU
            On exit db = dv*db
    
    @param[in]
    dv      COMPLEX array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    ********************************************************************/
extern "C" void
magmablas_cprbt_mv_q(
    magma_int_t n, 
    magmaFloatComplex *dv, magmaFloatComplex *db,
    magma_queue_t queue)
{
    magma_int_t threads = block_length;
    magma_int_t grid = magma_ceildiv( n, 2*block_length );

    magmablas_capply_vector_kernel<<< grid, threads, 0, queue >>>(n, dv, 0, db, 0);

    threads = block_length;
    grid = magma_ceildiv( n, 4*block_length );

    magmablas_capply_vector_kernel<<< grid, threads, 0, queue >>>(n/2, dv, n, db, 0);
    magmablas_capply_vector_kernel<<< grid, threads, 0, queue >>>(n/2, dv, n+n/2, db, n/2);
}

/**
    @see magmablas_cprbt_mtv_q
    ********************************************************************/
extern "C" void
magmablas_cprbt_mv(
    magma_int_t n, 
    magmaFloatComplex *dv, magmaFloatComplex *db)
{
    magmablas_cprbt_mv_q(n, dv, db, magma_stream);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    CPRBT randomize a square general matrix using partial randomized transformation
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns and rows of the matrix dA.  n >= 0.
    
    @param[in,out]
    dA      COMPLEX array, dimension (n,ldda)
            The n-by-n matrix dA
            On exit dA = duT*dA*d_V
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDA >= max(1,n).
    
    @param[in]
    du      COMPLEX array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix U
    
    @param[in]
    dv      COMPLEX array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    ********************************************************************/
extern "C" void 
magmablas_cprbt_q(
    magma_int_t n, 
    magmaFloatComplex *dA, magma_int_t ldda, 
    magmaFloatComplex *du, magmaFloatComplex *dv,
    magma_queue_t queue)
{
    du += ldda;
    dv += ldda;

    dim3 threads(block_height, block_width);
    dim3 grid( magma_ceildiv( n, 4*block_height ), 
               magma_ceildiv( n, 4*block_width  ));

    magmablas_celementary_multiplication_kernel<<< grid, threads, 0, queue >>>(n/2, dA,            0, ldda, du,   0, dv,   0);
    magmablas_celementary_multiplication_kernel<<< grid, threads, 0, queue >>>(n/2, dA,     ldda*n/2, ldda, du,   0, dv, n/2);
    magmablas_celementary_multiplication_kernel<<< grid, threads, 0, queue >>>(n/2, dA,          n/2, ldda, du, n/2, dv,   0);
    magmablas_celementary_multiplication_kernel<<< grid, threads, 0, queue >>>(n/2, dA, ldda*n/2+n/2, ldda, du, n/2, dv, n/2);

    dim3 threads2(block_height, block_width);
    dim3 grid2( magma_ceildiv( n, 2*block_height ), 
                magma_ceildiv( n, 2*block_width  ));
    magmablas_celementary_multiplication_kernel<<< grid2, threads2, 0, queue >>>(n, dA, 0, ldda, du, -ldda, dv, -ldda);
}


/**
    @see magmablas_cprbt_q
    ********************************************************************/
extern "C" void 
magmablas_cprbt(
    magma_int_t n, 
    magmaFloatComplex *dA, magma_int_t ldda, 
    magmaFloatComplex *du, magmaFloatComplex *dv)
{
    magmablas_cprbt_q(n, dA, ldda, du, dv, magma_stream);
}
