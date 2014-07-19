/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zgesellcmv.cu normal z -> c, Fri Jul 18 17:34:27 2014

*/
#include "cuda_runtime.h"
#include <stdio.h>
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


#define PRECISION_c


// SELLC SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
__global__ void 
cgesellcmv_kernel(   int num_rows, 
                     int num_cols,
                     int blocksize,
                     magmaFloatComplex alpha, 
                     magmaFloatComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaFloatComplex *d_x,
                     magmaFloatComplex beta, 
                     magmaFloatComplex *d_y)
{
    // threads assigned to rows
    int Idx = blockDim.x * blockIdx.x + threadIdx.x ;
    int offset = d_rowptr[ blockIdx.x ];
    int border = (d_rowptr[ blockIdx.x+1 ]-offset)/blocksize;
    if(Idx < num_rows ){
        magmaFloatComplex dot = MAGMA_C_MAKE(0.0, 0.0);
        for ( int n = 0; n < border; n++){ 
            int col = d_colind [offset+ blocksize * n + threadIdx.x ];
            magmaFloatComplex val = d_val[offset+ blocksize * n + threadIdx.x];
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
    alpha       magmaFloatComplex
                scalar multiplier

    @param
    d_val       magmaFloatComplex*
                array containing values of A in SELLC/P

    @param
    d_colind    magma_int_t*
                columnindices of A in SELLC/P

    @param
    d_rowptr    magma_int_t*
                rowpointer of SELLP

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
magma_cgesellcmv(   magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y ){



   // the kernel can only handle up to 65535 slices 
   // (~2M rows for blocksize 32)
   dim3 grid( slices, 1, 1);

   cgesellcmv_kernel<<< grid, blocksize, 0, magma_stream >>>
   ( m, n, blocksize, alpha,
        d_val, d_colind, d_rowptr, d_x, beta, d_y );

   return MAGMA_SUCCESS;
}

