/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zmgeelltmv.cu normal z -> d, Fri Jul 18 17:34:28 2014

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
dmgeelltmv_kernel( int num_rows, 
                 int num_cols,
                 int num_vecs,
                 int num_cols_per_row,
                 double alpha, 
                 double *d_val, 
                 magma_index_t *d_colind,
                 double *d_x,
                 double beta, 
                 double *d_y)
{
    extern __shared__ double dot[];
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++ )
                dot[ threadIdx.x+ i*blockDim.x ] = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = d_colind [ num_rows * n + row ];
            double val = d_val [ num_rows * n + row ];
            if( val != 0){
                for( int i=0; i<num_vecs; i++ )
                    dot[ threadIdx.x + i*blockDim.x ] += 
                                        val * d_x[col + i * num_cols ];
            }
        }
        for( int i=0; i<num_vecs; i++ )
                d_y[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] 
                                * alpha + beta * d_y [ row + i*num_cols ];
    }
}





/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is ELL. 
    
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
    num_vecs    mama_int_t
                number of vectors
                
    @param
    nnz_per_row magma_int_t
                number of elements in the longest row 
                
    @param
    alpha       double
                scalar multiplier

    @param
    d_val       double*
                array containing values of A in ELL

    @param
    d_colind    magma_int_t*
                columnindices of A in ELL

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
magma_dmgeelltmv(  magma_trans_t transA,
                   magma_int_t m, magma_int_t n,
                   magma_int_t num_vecs,
                   magma_int_t nnz_per_row,
                   double alpha,
                   double *d_val,
                   magma_index_t *d_colind,
                   double *d_x,
                   double beta,
                   double *d_y ){



    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                * sizeof( double ); // num_vecs vectors 
    dmgeelltmv_kernel<<< grid, BLOCK_SIZE, MEM_SIZE >>>
        ( m, n, num_vecs, nnz_per_row, alpha, d_val, d_colind, d_x, beta, d_y );


    return MAGMA_SUCCESS;
}



