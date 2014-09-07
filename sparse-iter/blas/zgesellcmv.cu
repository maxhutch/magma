/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/
#include "cuda_runtime.h"
#include <stdio.h>
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


#define PRECISION_z


// SELLC SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
__global__ void 
zgesellcmv_kernel(   int num_rows, 
                     int num_cols,
                     int blocksize,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
    // threads assigned to rows
    int Idx = blockDim.x * blockIdx.x + threadIdx.x ;
    int offset = d_rowptr[ blockIdx.x ];
    int border = (d_rowptr[ blockIdx.x+1 ]-offset)/blocksize;
    if(Idx < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < border; n++){ 
            int col = d_colind [offset+ blocksize * n + threadIdx.x ];
            magmaDoubleComplex val = d_val[offset+ blocksize * n + threadIdx.x];
            if( val != 0){
                  dot=dot+val*d_x[col];
            }
        }

        d_y[ Idx ] = dot * alpha + beta * d_y [ Idx ];
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLC/SELLP.
    
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
    blocksize   magma_int_t
                number of rows in one ELL-slice

    @param
    slices      magma_int_t
                number of slices in matrix

    @param
    alignment   magma_int_t
                number of threads assigned to one row (=1)

    @param
    alpha       magmaDoubleComplex
                scalar multiplier

    @param
    d_val       magmaDoubleComplex*
                array containing values of A in SELLC/P

    @param
    d_colind    magma_int_t*
                columnindices of A in SELLC/P

    @param
    d_rowptr    magma_int_t*
                rowpointer of SELLP

    @param
    d_x         magmaDoubleComplex*
                input vector x

    @param
    beta        magmaDoubleComplex
                scalar multiplier

    @param
    d_y         magmaDoubleComplex*
                input/output vector y


    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zgesellcmv(   magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){



   // the kernel can only handle up to 65535 slices 
   // (~2M rows for blocksize 32)
   dim3 grid( slices, 1, 1);

   zgesellcmv_kernel<<< grid, blocksize, 0, magma_stream >>>
   ( m, n, blocksize, alpha,
        d_val, d_colind, d_rowptr, d_x, beta, d_y );

   return MAGMA_SUCCESS;
}

