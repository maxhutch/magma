/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zgeellmv.cu normal z -> d, Fri Jul 18 17:34:27 2014

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


// ELLPACK SpMV kernel
//Michael Garland
__global__ void 
dgeellmv_kernel( int num_rows, 
                 int num_cols,
                 int num_cols_per_row,
                 double alpha, 
                 double *d_val, 
                 magma_index_t *d_colind,
                 double *d_x,
                 double beta, 
                 double *d_y)
{
int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_rows ){
        double dot = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = d_colind [ num_cols_per_row * row + n ];
            double val = d_val [ num_cols_per_row * row + n ];
            if( val != 0)
                dot += val * d_x[col ];
        }
        d_y[ row ] = dot * alpha + beta * d_y [ row ];
    }
}

// shifted ELLPACK SpMV kernel
//Michael Garland
__global__ void 
dgeellmv_kernel_shift( int num_rows, 
                       int num_cols,
                       int num_cols_per_row,
                       double alpha, 
                       double lambda, 
                       double *d_val, 
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta, 
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       double *d_y)
{
int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_rows ){
        double dot = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = d_colind [ num_cols_per_row * row + n ];
            double val = d_val [ num_cols_per_row * row + n ];
            if( val != 0)
                dot += val * d_x[col ];
        }
        if( row<blocksize )
            d_y[ row ] = dot * alpha - lambda * d_x[ offset+row ] + beta * d_y [ row ];
        else
            d_y[ row ] = dot * alpha - lambda * d_x[ add_rows[row-blocksize] ] + beta * d_y [ row ];   
    }
}





/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLPACK.
    
    Arguments
    ---------
    
    @param
    transA      magma_trans_t
                transposition parameter for A
                
    @param
    m           magma_int_t
                number of rows in A

    @param
    n           magma_int_t
                number of columns in A 
                
    @param
    nnz_per_row magma_int_t
                number of elements in the longest row 

    @param
    alpha       double
                scalar multiplier

    @param
    d_val       double*
                array containing values of A in ELLPACK

    @param
    d_colind    magma_int_t*
                columnindices of A in ELLPACK

    @param
    d_x         double*
                input vector x

    @param
    beta        double
                scalar multiplier

    @param
    d_y         double*
                input/output vector y


    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" magma_int_t
magma_dgeellmv(magma_trans_t transA,
               magma_int_t m, magma_int_t n,
               magma_int_t nnz_per_row,
               double alpha,
               double *d_val,
               magma_index_t *d_colind,
               double *d_x,
               double beta,
               double *d_y ){



   dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   dgeellmv_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( m, n, nnz_per_row, alpha, d_val, d_colind, d_x, beta, d_y );


   return MAGMA_SUCCESS;
}



/**
    Purpose
    -------
    
    This routine computes y = alpha *( A - lambda I ) * x + beta * y on the GPU.
    Input format is ELLPACK.
    It is the shifted version of the ELLPACK SpMV.
    
    Arguments
    ---------
    
    @param
    transA      magma_trans_t
                transposition parameter for A

    @param
    m           magma_int_t
                number of rows in A

    @param
    n           magma_int_t
                number of columns in A 
    @param
    nnz_per_row magma_int_t
                number of elements in the longest row 
                
    @param
    alpha       double
                scalar multiplier
                
    @param
    lambda      double
                scalar multiplier

    @param
    d_val       double*
                array containing values of A in ELLPACK

    @param
    d_colind    magma_int_t*
                columnindices of A in ELLPACK

    @param
    d_x         double*
                input vector x

    @param
    beta        double
                scalar multiplier
                
    @param
    offset      magma_int_t 
                in case not the main diagonal is scaled
                
    @param
    blocksize   magma_int_t 
                in case of processing multiple vectors  
                
    @param
    add_rows    magma_int_t*
                in case the matrixpowerskernel is used

    @param
    d_y         double*
                input/output vector y


    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" magma_int_t
magma_dgeellmv_shift( magma_trans_t transA,
                      magma_int_t m, magma_int_t n,
                      magma_int_t nnz_per_row,
                      double alpha,
                      double lambda,
                      double *d_val,
                      magma_index_t *d_colind,
                      double *d_x,
                      double beta,
                      int offset,
                      int blocksize,
                      magma_index_t *add_rows,
                      double *d_y ){



   dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   dgeellmv_kernel_shift<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( m, n, nnz_per_row, alpha, lambda, d_val, d_colind, d_x, 
                                    beta, offset, blocksize, add_rows, d_y );


   return MAGMA_SUCCESS;
}



