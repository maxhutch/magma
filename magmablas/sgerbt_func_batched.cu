/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgerbt_func_batched.cu normal z -> s, Fri Jan 30 19:00:10 2015

       @author Adrien Remy
       @author Azzam Haidar
*/
#include "common_magma.h"
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
magmablas_sprbt_mtv_batched(
    magma_int_t n, 
    float *du, float **db_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    /*

     */
    magma_int_t threads = block_length;
    dim3 grid ( n/(4*block_length) + ((n%(4*block_length))!=0), batchCount);

    magmablas_sapply_transpose_vector_kernel_batched<<< grid, threads, 0, queue >>>(n/2, du, n, db_array, 0);
    magmablas_sapply_transpose_vector_kernel_batched<<< grid, threads, 0, queue >>>(n/2, du, n+n/2, db_array, n/2);

    threads = block_length;
    grid = n/(2*block_length) + ((n%(2*block_length))!=0);
    magmablas_sapply_transpose_vector_kernel_batched<<< grid, threads, 0, queue >>>(n, du, 0, db_array, 0);
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
magmablas_sprbt_mv_batched(
    magma_int_t n, 
    float *dv, float **db_array,
    magma_int_t batchCount, magma_queue_t queue)
{

    magma_int_t threads = block_length;
    dim3 grid ( n/(2*block_length) + ((n%(2*block_length))!=0), batchCount);

    magmablas_sapply_vector_kernel_batched<<< grid, threads, 0, queue >>>(n, dv, 0, db_array, 0);


    threads = block_length;
    grid = n/(4*block_length) + ((n%(4*block_length))!=0);

    magmablas_sapply_vector_kernel_batched<<< grid, threads, 0, queue >>>(n/2, dv, n, db_array, 0);
    magmablas_sapply_vector_kernel_batched<<< grid, threads, 0, queue >>>(n/2, dv, n+n/2, db_array, n/2);


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
magmablas_sprbt_batched(
    magma_int_t n, 
    float **dA_array, magma_int_t ldda, 
    float *du, float *dv,
    magma_int_t batchCount, magma_queue_t queue)
{
    du += ldda;
    dv += ldda;

    dim3 threads(block_height, block_width);
    dim3 grid(n/(4*block_height) + ((n%(4*block_height))!=0), 
            n/(4*block_width)  + ((n%(4*block_width))!=0),
            batchCount);

    magmablas_selementary_multiplication_kernel_batched<<< grid, threads, 0, queue >>>(n/2, dA_array,            0, ldda, du,   0, dv,   0);
    magmablas_selementary_multiplication_kernel_batched<<< grid, threads, 0, queue >>>(n/2, dA_array,     ldda*n/2, ldda, du,   0, dv, n/2);
    magmablas_selementary_multiplication_kernel_batched<<< grid, threads, 0, queue >>>(n/2, dA_array,          n/2, ldda, du, n/2, dv,   0);
    magmablas_selementary_multiplication_kernel_batched<<< grid, threads, 0, queue >>>(n/2, dA_array, ldda*n/2+n/2, ldda, du, n/2, dv, n/2);

    dim3 threads2(block_height, block_width);
    dim3 grid2(n/(2*block_height) + ((n%(2*block_height))!=0), 
            n/(2*block_width)  + ((n%(2*block_width))!=0),
            batchCount);
    magmablas_selementary_multiplication_kernel_batched<<< grid2, threads2, 0, queue >>>(n, dA_array, 0, ldda, du, -ldda, dv, -ldda);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


