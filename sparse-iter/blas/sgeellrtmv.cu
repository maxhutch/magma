/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zgeellrtmv.cu normal z -> s, Fri May 30 10:41:37 2014

*/

#include "common_magma.h"

//F. Vázquez, G. Ortega, J.J. Fernández, E.M. Garzón, Almeria University
__global__ void 
sgeellrtmv_kernel_32( int num_rows, 
                 int num_cols,
                 float alpha, 
                 float *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 float *d_x,
                 float beta, 
                 float *d_y,
                 int T,
                 int alignment )
{
int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ float shared[];

    if(i < num_rows ){
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //float val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            float val = d_val[ k*(T)+(i*alignment)+idp ];
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
sgeellrtmv_kernel_16( int num_rows, 
                 int num_cols,
                 float alpha, 
                 float *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 float *d_x,
                 float beta, 
                 float *d_y,
                 int T,
                 int alignment )
{
int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ float shared[];

    if(i < num_rows ){
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //float val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            float val = d_val[ k*(T)+(i*alignment)+idp ];
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
sgeellrtmv_kernel_8( int num_rows, 
                 int num_cols,
                 float alpha, 
                 float *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 float *d_x,
                 float beta, 
                 float *d_y,
                 int T,
                 int alignment )
{
int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ float shared[];

    if(i < num_rows ){
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //float val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            float val = d_val[ k*(T)+(i*alignment)+idp ];
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



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLRT. The ideas are taken from 
    "Improving the performance of the sparse matrix
    vector product with GPUs", (CIT 2010), 
    and modified to provide correct values.

    
    Arguments
    =========
    const char *transA                  transpose info for matrix (not needed)
    magma_int_t m                       number of rows 
    magma_int_t n                       number of columns
    magma_int_t nnz_per_row             max number of nonzeros in a row
    float alpha            scalar alpha
    float *d_val           val array
    magma_int_t *d_colind               col indices  
    magma_int_t *d_rowlength            number of elements in each row
    float *d_x             input vector x
    float beta             scalar beta
    float *d_y             output vector y
    magma_int_t blocksize               threads per block
    magma_int_t alignment               threads assigned to each row

    =====================================================================    */

extern "C" magma_int_t
magma_sgeellrtmv(  magma_trans_t transA,
                   magma_int_t m, magma_int_t n,
                   magma_int_t nnz_per_row,
                   float alpha,
                   float *d_val,
                   magma_index_t *d_colind,
                   magma_index_t *d_rowlength,
                   float *d_x,
                   float beta,
                   float *d_y,
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

    int Ms = alignment * blocksize * sizeof( float );
    // printf("launch kernel: %dx%d %d %d\n", grid.x, grid.y, num_threads , Ms);

    if( alignment == 32 ){
        sgeellrtmv_kernel_32<<< grid, num_threads , Ms, magma_stream >>>
                 ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                 alignment, real_row_length );
    }
    else if( alignment == 16 ){
        sgeellrtmv_kernel_16<<< grid, num_threads , Ms, magma_stream >>>
                 ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                 alignment, real_row_length );
    }
    else if( alignment == 8 ){
        sgeellrtmv_kernel_8<<< grid, num_threads , Ms, magma_stream >>>
                 ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                 alignment, real_row_length );
    }
    else{
        printf("error: alignment %d not supported.\n", alignment);
        exit(-1);
    }



   return MAGMA_SUCCESS;
}


