/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zgesellcmv.cu normal z -> c, Fri May 30 10:41:36 2014

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


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLC/SELLP.
    
    Arguments
    =========

    magma_trans_t transA            transpose A?
    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magma_int_t blocksize           number of rows in one ELL-slice
    magma_int_t slices              number of slices in matrix
    magma_int_t alignment           number of threads assigned to one row (=1)
    magmaFloatComplex alpha        scalar multiplier
    magmaFloatComplex *d_val       array containing values of A in SELLC/P
    magma_int_t *d_colind           columnindices of A in SELLC/P
    magma_int_t *d_rowptr           rowpointer of SELLP
    magmaFloatComplex *d_x         input vector x
    magmaFloatComplex beta         scalar multiplier
    magmaFloatComplex *d_y         input/output vector y

    ======================================================================    */

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

