/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
zvjacobisetup_gpu(  int num_rows, 
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *c,
                    magmaDoubleComplex *x){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        c[row] = b[row] / d[row];
        x[row] = c[row];
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

    @param
    num_rows    magma_int_t
                number of rows
                
    @param
    b           magma_z_vector
                RHS b

    @param
    d           magma_z_vector
                vector with diagonal entries

    @param
    c           magma_z_vector*
                c = D^(-1) * b

    @param
    x           magma_z_vector*
                iteration vector

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_vector_gpu(  int num_rows, 
                                magmaDoubleComplex *b, 
                                magmaDoubleComplex *d, 
                                magmaDoubleComplex *c,
                                magmaDoubleComplex *x ){


   dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   zvjacobisetup_gpu<<< grid, BLOCK_SIZE, 0 >>>( num_rows, b, d, c, x );

   return MAGMA_SUCCESS;
}






__global__ void 
zjacobidiagscal_kernel(  int num_rows, 
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *c){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        c[row] = b[row] * d[row];
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

    @param
    num_rows    magma_int_t
                number of rows
                
    @param
    b           magma_z_vector
                RHS b

    @param
    d           magma_z_vector
                vector with diagonal entries

    @param
    c           magma_z_vector*
                c = D^(-1) * b

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobi_diagscal(         int num_rows, 
                                magmaDoubleComplex *b, 
                                magmaDoubleComplex *d, 
                                magmaDoubleComplex *c){


   dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   zjacobidiagscal_kernel<<< grid, BLOCK_SIZE, 0 >>>( num_rows, b, d, c );

   return MAGMA_SUCCESS;
}



