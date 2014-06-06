/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zjacobisetup.cu normal z -> c, Fri May 30 10:41:37 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
cvjacobisetup_gpu(  int num_rows, 
                    magmaFloatComplex *b, 
                    magmaFloatComplex *d, 
                    magmaFloatComplex *c){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        c[row] = b[row] / d[row];
    }
}





/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    =========

    magma_c_vector b                          RHS b
    magma_c_vector d                          vector with diagonal entries
    magma_c_vector *c                         c = D^(-1) * b

    ========================================================================  */

extern "C" magma_int_t
magma_cjacobisetup_vector_gpu(  int num_rows, 
                                magmaFloatComplex *b, 
                                magmaFloatComplex *d, 
                                magmaFloatComplex *c){


   dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   cvjacobisetup_gpu<<< grid, BLOCK_SIZE, 0 >>>( num_rows, b, d, c );

   return MAGMA_SUCCESS;
}






__global__ void 
cjacobidiagscal_kernel(  int num_rows, 
                    magmaFloatComplex *b, 
                    magmaFloatComplex *d, 
                    magmaFloatComplex *c){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        c[row] = b[row] * d[row];
    }
}





/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    =========

    magma_c_vector b                          RHS b
    magma_c_vector d                          vector with diagonal entries
    magma_c_vector *c                         c = D^(-1) * b

    ========================================================================  */

extern "C" magma_int_t
magma_cjacobi_diagscal(         int num_rows, 
                                magmaFloatComplex *b, 
                                magmaFloatComplex *d, 
                                magmaFloatComplex *c){


   dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   cjacobidiagscal_kernel<<< grid, BLOCK_SIZE, 0 >>>( num_rows, b, d, c );

   return MAGMA_SUCCESS;
}



