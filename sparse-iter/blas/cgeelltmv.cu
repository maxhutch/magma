/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zgeelltmv.cu normal z -> c, Fri May 30 10:41:36 2014

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


// ELL SpMV kernel
//Michael Garland
__global__ void 
cgeelltmv_kernel( int num_rows, 
                 int num_cols,
                 int num_cols_per_row,
                 magmaFloatComplex alpha, 
                 magmaFloatComplex *d_val, 
                 magma_index_t *d_colind,
                 magmaFloatComplex *d_x,
                 magmaFloatComplex beta, 
                 magmaFloatComplex *d_y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_rows ){
        magmaFloatComplex dot = MAGMA_C_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = d_colind [ num_rows * n + row ];
            magmaFloatComplex val = d_val [ num_rows * n + row ];
            if( val != 0)
                dot += val * d_x[col ];
        }
        d_y[ row ] = dot * alpha + beta * d_y [ row ];
    }
}

// shifted ELL SpMV kernel
//Michael Garland
__global__ void 
cgeelltmv_kernel_shift( int num_rows, 
                        int num_cols,
                        int num_cols_per_row,
                        magmaFloatComplex alpha, 
                        magmaFloatComplex lambda, 
                        magmaFloatComplex *d_val, 
                        magma_index_t *d_colind,
                        magmaFloatComplex *d_x,
                        magmaFloatComplex beta, 
                        int offset,
                        int blocksize,
                        magma_index_t *add_rows,
                        magmaFloatComplex *d_y){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_rows ){
        magmaFloatComplex dot = MAGMA_C_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = d_colind [ num_rows * n + row ];
            magmaFloatComplex val = d_val [ num_rows * n + row ];
            if( val != 0)
                dot += val * d_x[col ];
        }
        if( row<blocksize )
            d_y[ row ] = dot * alpha - lambda 
                    * d_x[ offset+row ] + beta * d_y [ row ];
        else
            d_y[ row ] = dot * alpha - lambda 
                    * d_x[ add_rows[row-blocksize] ] + beta * d_y [ row ];            
    }
}




/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is ELL.
    
    Arguments
    =========

    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magmaFloatComplex alpha        scalar multiplier
    magmaFloatComplex *d_val       array containing values of A in ELL
    magma_int_t *d_colind           columnindices of A in ELL
    magmaFloatComplex *d_x         input vector x
    magmaFloatComplex beta         scalar multiplier
    magmaFloatComplex *d_y         input/output vector y

    ======================================================================    */

extern "C" magma_int_t
magma_cgeelltmv(   magma_trans_t transA,
                   magma_int_t m, magma_int_t n,
                   magma_int_t nnz_per_row,
                   magmaFloatComplex alpha,
                   magmaFloatComplex *d_val,
                   magma_index_t *d_colind,
                   magmaFloatComplex *d_x,
                   magmaFloatComplex beta,
                   magmaFloatComplex *d_y ){



   dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   cgeelltmv_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( m, n, nnz_per_row, alpha, d_val, d_colind, d_x, beta, d_y );


   return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======
    
    This routine computes y = alpha *( A - lambda I ) * x + beta * y on the GPU.
    Input format is ELL.
    
    Arguments
    =========

    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magmaFloatComplex alpha        scalar multiplier
    magmaFloatComplex lambda       scalar multiplier
    magmaFloatComplex *d_val       array containing values of A in ELL
    magma_int_t *d_colind           columnindices of A in ELL
    magmaFloatComplex *d_x         input vector x
    magmaFloatComplex beta         scalar multiplier
    magmaFloatComplex *d_y         input/output vector y

    ======================================================================    */

extern "C" magma_int_t
magma_cgeelltmv_shift( magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex lambda,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaFloatComplex *d_y ){



   dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
   magmaFloatComplex tmp_shift;
   //magma_csetvector(1,&lambda,1,&tmp_shift,1); 
   tmp_shift = lambda;
   cgeelltmv_kernel_shift<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( m, n, nnz_per_row, alpha, tmp_shift, d_val, d_colind, d_x, 
                            beta, offset, blocksize, add_rows, d_y );


   return MAGMA_SUCCESS;
}



