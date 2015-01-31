/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zjacobisetup.cu normal z -> s, Fri Jan 30 19:00:29 2015
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
svjacobisetup_gpu(  int num_rows, 
                    int num_vecs,
                    float *b, 
                    float *d, 
                    float *c,
                    float *x){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++ ){
            c[row+i*num_rows] = b[row+i*num_rows] / d[row];
            x[row+i*num_rows] = c[row+i*num_rows];
        }
    }
}





/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    b           magma_s_vector
                RHS b

    @param[in]
    d           magma_s_vector
                vector with diagonal entries

    @param[out]
    c           magma_s_vector*
                c = D^(-1) * b

    @param[out]
    x           magma_s_vector*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobisetup_vector_gpu(
    int num_rows, 
    magma_s_vector b, 
    magma_s_vector d, 
    magma_s_vector c,
    magma_s_vector *x,
    magma_queue_t queue )
{
    dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
   int num_vecs = b.num_rows / num_rows;
    magma_int_t threads = BLOCK_SIZE;
   svjacobisetup_gpu<<< grid, threads, 0 >>>
                ( num_rows, num_vecs, b.dval, d.dval, c.dval, x->val );

   return MAGMA_SUCCESS;
}






__global__ void 
sjacobidiagscal_kernel(  int num_rows,
                         int num_vecs, 
                    float *b, 
                    float *d, 
                    float *c){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++)
            c[row+i*num_rows] = b[row+i*num_rows] * d[row];
    }
}





/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    b           magma_s_vector
                RHS b

    @param[in]
    d           magma_s_vector
                vector with diagonal entries

    @param[out]
    c           magma_s_vector*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobi_diagscal(
    int num_rows, 
    magma_s_vector d, 
    magma_s_vector b, 
    magma_s_vector *c,
    magma_queue_t queue )
{
    dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
   int num_vecs = b.num_rows/num_rows;
    magma_int_t threads = BLOCK_SIZE;
   sjacobidiagscal_kernel<<< grid, threads, 0 >>>( num_rows, num_vecs, b.dval, d.dval, c->val );

   return MAGMA_SUCCESS;
}



