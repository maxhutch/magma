/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zgecsrmv.cu normal z -> c, Fri Jul 18 17:34:27 2014

*/
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 256
#else
   #define BLOCK_SIZE 256
#endif


// CSR-SpMV kernel
__global__ void 
cgecsrmv_kernel( int num_rows, int num_cols, 
                 magmaFloatComplex alpha, 
                 magmaFloatComplex *d_val, 
                 magma_index_t *d_rowptr, 
                 magma_index_t *d_colind,
                 magmaFloatComplex *d_x,
                 magmaFloatComplex beta, 
                 magmaFloatComplex *d_y){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaFloatComplex dot = MAGMA_C_ZERO;
        int start = d_rowptr[ row ];
        int end = d_rowptr[ row+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * d_x[ d_colind[j] ];
        d_y[ row ] =  dot *alpha + beta * d_y[ row ];
    }
}

// shifted CSR-SpMV kernel
__global__ void 
cgecsrmv_kernel_shift( int num_rows, int num_cols, 
                       magmaFloatComplex alpha, 
                       magmaFloatComplex lambda, 
                       magmaFloatComplex *d_val, 
                       magma_index_t *d_rowptr, 
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta, 
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaFloatComplex *d_y){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaFloatComplex dot = MAGMA_C_ZERO;
        int start = d_rowptr[ row ];
        int end = d_rowptr[ row+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * d_x[ d_colind[j] ];
        if( row<blocksize )
            d_y[ row ] = dot * alpha - lambda 
                        * d_x[ offset+row ] + beta * d_y [ row ];
        else
            d_y[ row ] = dot * alpha - lambda 
                        * d_x[ add_rows[row-blocksize] ] + beta * d_y [ row ];   
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    The input format is CSR (val, row, col).
    
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
    alpha       magmaFloatComplex
                scalar multiplier

    @param
    d_val       magmaFloatComplex*
                array containing values of A in CSR

    @param
    d_rowptr    magma_int_t*
                rowpointer of A in CSR

    @param
    d_colind    magma_int_t*
                columnindices of A in CSR

    @param
    d_x         magmaFloatComplex*
                input vector x

    @param
    beta        magmaFloatComplex
                scalar multiplier

    @param
    d_y         magmaFloatComplex*
                input/output vector y


    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_cgecsrmv(     magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_rowptr,
                    magma_index_t *d_colind,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y ){

    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

    cgecsrmv_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                    (m, n, alpha, d_val, d_rowptr, d_colind, d_x, beta, d_y);

    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------
    
    This routine computes y = alpha * ( A -lambda I ) * x + beta * y on the GPU.
    It is a shifted version of the CSR-SpMV.
    
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
    alpha       magmaFloatComplex
                scalar multiplier

    @param
    lambda      magmaFloatComplex
                scalar multiplier

    @param
    d_val       magmaFloatComplex*
                array containing values of A in CSR

    @param
    d_rowptr    magma_int_t*
                rowpointer of A in CSR

    @param
    d_colind    magma_int_t*
                columnindices of A in CSR

    @param
    d_x         magmaFloatComplex*
                input vector x

    @param
    beta        magmaFloatComplex
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
    d_y         magmaFloatComplex*
                output vector y  

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_cgecsrmv_shift( magma_trans_t transA,
                      magma_int_t m, magma_int_t n,
                      magmaFloatComplex alpha,
                      magmaFloatComplex lambda,
                      magmaFloatComplex *d_val,
                      magma_index_t *d_rowptr,
                      magma_index_t *d_colind,
                      magmaFloatComplex *d_x,
                      magmaFloatComplex beta,
                      int offset,
                      int blocksize,
                      magma_index_t *add_rows,
                      magmaFloatComplex *d_y ){

    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

    cgecsrmv_kernel_shift<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                         (m, n, alpha, lambda, d_val, d_rowptr, d_colind, d_x, 
                                    beta, offset, blocksize, add_rows, d_y);

    return MAGMA_SUCCESS;
}



