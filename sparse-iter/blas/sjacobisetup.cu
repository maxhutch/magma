/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zjacobisetup.cu normal z -> s, Sun May  3 11:22:58 2015
       @author Hartwig Anzt

*/
#include "common_magmasparse.h"

#define BLOCK_SIZE 128


#define PRECISION_s

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
    b           magma_s_matrix
                RHS b

    @param[in]
    d           magma_s_matrix
                vector with diagonal entries

    @param[out]
    c           magma_s_matrix*
                c = D^(-1) * b

    @param[out]
    x           magma_s_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobisetup_vector_gpu(
    int num_rows, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix c,
    magma_s_matrix *x,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( num_rows, BLOCK_SIZE ) );
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
    b           magma_s_matrix
                RHS b

    @param[in]
    d           magma_s_matrix
                vector with diagonal entries

    @param[out]
    c           magma_s_matrix*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobi_diagscal(
    int num_rows, 
    magma_s_matrix d, 
    magma_s_matrix b, 
    magma_s_matrix *c,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( num_rows, BLOCK_SIZE ));
    int num_vecs = b.num_rows*b.num_cols/num_rows;
    magma_int_t threads = BLOCK_SIZE;
    sjacobidiagscal_kernel<<< grid, threads, 0 >>>( num_rows, num_vecs, b.dval, d.dval, c->val );

    return MAGMA_SUCCESS;
}













__global__ void 
sjacobiupdate_kernel(  int num_rows,
                       int num_cols, 
                    float *t, 
                    float *b, 
                    float *d, 
                    float *x){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        for( int i=0; i<num_cols; i++)
            x[row+i*num_rows] += (b[row+i*num_rows]-t[row+i*num_rows]) * d[row];
    }
}


/**
    Purpose
    -------

    Updates the iteration vector x for the Jacobi iteration
    according to
        x=x+d.*(b-t)
    where d is the diagonal of the system matrix A and t=Ax.

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    num_cols    magma_int_t
                number of cols
                
    @param[in]
    t           magma_s_matrix
                t = A*x
                
    @param[in]
    b           magma_s_matrix
                RHS b
                
    @param[in]
    d           magma_s_matrix
                vector with diagonal entries

    @param[out]
    x           magma_s_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobiupdate(
    magma_s_matrix t, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix *x,
    magma_queue_t queue )
{

    dim3 grid( magma_ceildiv( t.num_rows, BLOCK_SIZE ));
    magma_int_t threads = BLOCK_SIZE;
    sjacobiupdate_kernel<<< grid, threads, 0 >>>( t.num_rows, t.num_cols, t.dval, b.dval, d.dval, x->dval );

    return MAGMA_SUCCESS;
}










__global__ void 
sjacobispmvupdate_kernel(  
    int num_rows,
    int num_cols, 
    float * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    float *t, 
    float *b, 
    float *d, 
    float *x ){



    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    int j;

    if(row<num_rows){
        float dot = MAGMA_S_ZERO;
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        for( int i=0; i<num_cols; i++){
            for( j=start; j<end; j++){
                dot += dval[ j ] * x[ dcolind[j]+i*num_rows ];
            }
            x[row+i*num_rows] += (b[row+i*num_rows]-dot) * d[row];
        }
    }
}





/**
    Purpose
    -------

    Updates the iteration vector x for the Jacobi iteration
    according to
        x=x+d.*(b-Ax)


    Arguments
    ---------

    @param[in]
    maxiter     magma_int_t
                number of Jacobi iterations   
                
    @param[in]
    A           magma_s_matrix
                system matrix
                
    @param[in]
    t           magma_s_matrix
                workspace
                
    @param[in]
    b           magma_s_matrix
                RHS b
                
    @param[in]
    d           magma_s_matrix
                vector with diagonal entries

    @param[out]
    x           magma_s_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobispmvupdate(
    magma_int_t maxiter,
    magma_s_matrix A,
    magma_s_matrix t, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix *x,
    magma_queue_t queue )
{

    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE;
    dim3 grid( magma_ceildiv( t.num_rows, BLOCK_SIZE ));
    magma_int_t threads = BLOCK_SIZE;

    for( magma_int_t i=0; i<maxiter; i++ ) {
        // distinct routines imply synchronization
        // magma_s_spmv( c_one, A, *x, c_zero, t, queue );                // t =  A * x
        // sjacobiupdate_kernel<<< grid, threads, 0 >>>( t.num_rows, t.num_cols, t.dval, b.dval, d.dval, x->dval );
        // merged in one implies asynchronous update
        sjacobispmvupdate_kernel<<< grid, threads, 0 >>>
            ( t.num_rows, t.num_cols, A.dval, A.drow, A.dcol, t.dval, b.dval, d.dval, x->dval );

    }

    return MAGMA_SUCCESS;
}










