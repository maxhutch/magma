/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/

#include "common_magma.h"

//F. Vázquez, G. Ortega, J.J. Fernández, E.M. Garzón, Almeria University
__global__ void 
zgeellrtmv_kernel_32( int num_rows, 
                 int num_cols,
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y,
                 int T,
                 int alignment )
{
int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d_x[ col ];
        }
        shared[idb]  = dot;
        if( idp < 16 ){
            shared[idb]+=shared[idb+16];
            if( idp < 8 ) shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                d_y[i] = (shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
            }

        }
    }

}

//F. Vázquez, G. Ortega, J.J. Fernández, E.M. Garzón, Almeria University
__global__ void 
zgeellrtmv_kernel_16( int num_rows, 
                 int num_cols,
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y,
                 int T,
                 int alignment )
{
int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d_x[ col ];
        }
        shared[idb]  = dot;
        if( idp < 8 ){
            shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                d_y[i] = (shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
            }

        }
    }

}

//F. Vázquez, G. Ortega, J.J. Fernández, E.M. Garzón, Almeria University
__global__ void 
zgeellrtmv_kernel_8( int num_rows, 
                 int num_cols,
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y,
                 int T,
                 int alignment )
{
int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d_x[ col ];
        }
        shared[idb]  = dot;
        if( idp < 4 ){
            shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                d_y[i] = (shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
            }

        }
    }

}



/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLRT. The ideas are taken from 
    "Improving the performance of the sparse matrix
    vector product with GPUs", (CIT 2010), 
    and modified to provide correct values.

    
    Arguments
    ---------

    @param
    transA      magma_trans_t
                transposition parameter for A
    @param
    m           magma_int_t
                number of rows 

    @param
    n           magma_int_t
                number of columns

    @param
    nnz_per_row magma_int_t
                max number of nonzeros in a row

    @param
    alpha       magmaDoubleComplex
                scalar alpha

    @param
    d_val       magmaDoubleComplex*
                val array

    @param
    d_colind    magma_int_t*
                col indices  

    @param
    d_rowlength magma_int_t*
                number of elements in each row

    @param
    d_x         magmaDoubleComplex*
                input vector x

    @param
    beta        magmaDoubleComplex
                scalar beta

    @param
    d_y         magmaDoubleComplex*
                output vector y

    @param
    blocksize   magma_int_t
                threads per block

    @param
    alignment   magma_int_t
                threads assigned to each row


    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zgeellrtmv(  magma_trans_t transA,
                   magma_int_t m, magma_int_t n,
                   magma_int_t nnz_per_row,
                   magmaDoubleComplex alpha,
                   magmaDoubleComplex *d_val,
                   magma_index_t *d_colind,
                   magma_index_t *d_rowlength,
                   magmaDoubleComplex *d_x,
                   magmaDoubleComplex beta,
                   magmaDoubleComplex *d_y,
                   magma_int_t alignment,
                   magma_int_t blocksize ){


    int num_blocks = ( (m+blocksize-1)/blocksize);

    int num_threads = alignment*blocksize;

    int real_row_length = ((int)(nnz_per_row+alignment-1)/alignment)
                            *alignment;

    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 && num_threads > 256 )
        printf("error: too much shared memory requested.\n");

    int dimgrid1 = sqrt(num_blocks);
    int dimgrid2 = (num_blocks + dimgrid1 -1 ) / dimgrid1;
    dim3 grid( dimgrid1, dimgrid2, 1);

    int Ms = alignment * blocksize * sizeof( magmaDoubleComplex );
    // printf("launch kernel: %dx%d %d %d\n", grid.x, grid.y, num_threads , Ms);

    if( alignment == 32 ){
        zgeellrtmv_kernel_32<<< grid, num_threads , Ms, magma_stream >>>
                 ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                 alignment, real_row_length );
    }
    else if( alignment == 16 ){
        zgeellrtmv_kernel_16<<< grid, num_threads , Ms, magma_stream >>>
                 ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                 alignment, real_row_length );
    }
    else if( alignment == 8 ){
        zgeellrtmv_kernel_8<<< grid, num_threads , Ms, magma_stream >>>
                 ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                 alignment, real_row_length );
    }
    else{
        printf("error: alignment %d not supported.\n", alignment);
        exit(-1);
    }



   return MAGMA_SUCCESS;
}


