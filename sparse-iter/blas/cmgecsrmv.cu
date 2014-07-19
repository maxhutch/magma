/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zmgecsrmv.cu normal z -> c, Fri Jul 18 17:34:28 2014

*/
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
cmgecsrmv_kernel( int num_rows, int num_cols, 
                  int num_vecs,
                  magmaFloatComplex alpha, 
                  magmaFloatComplex *d_val, 
                  magma_index_t *d_rowptr, 
                  magma_index_t *d_colind,
                  magmaFloatComplex *d_x,
                  magmaFloatComplex beta, 
                  magmaFloatComplex *d_y){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;
    extern __shared__ magmaFloatComplex dot[];

    if( row<num_rows ){
        for( int i=0; i<num_vecs; i++ )
                dot[ threadIdx.x+ i*blockDim.x ] = MAGMA_C_MAKE(0.0, 0.0);
        int start = d_rowptr[ row ] ;
        int end = d_rowptr[ row+1 ];
        for( j=start; j<end; j++ ){
            int col = d_colind [ j ];
            magmaFloatComplex val = d_val[ j ];
            for( int i=0; i<num_vecs; i++ )
                dot[ threadIdx.x + i*blockDim.x ] += 
                                    val * d_x[ col + i*num_cols ];
        }
        for( int i=0; i<num_vecs; i++ )
            d_y[ row +i*num_cols ] = alpha * dot[ threadIdx.x + i*blockDim.x ] 
                                             + beta * d_y[ row + i*num_cols ];
    }
}



/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is CSR. 
    
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
magma_cmgecsrmv(    magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs, 
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_rowptr,
                    magma_index_t *d_colind,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y ){

    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                    * sizeof( magmaFloatComplex ); // num_vecs vectors 
    cmgecsrmv_kernel<<< grid, BLOCK_SIZE, MEM_SIZE >>>
            (m, n, num_vecs, alpha, d_val, d_rowptr, d_colind, d_x, beta, d_y);

   return MAGMA_SUCCESS;
}



