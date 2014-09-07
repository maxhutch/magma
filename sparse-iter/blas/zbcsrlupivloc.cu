/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h


#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include "magma.h"


#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif

#define PRECISION_z

#define  Ablockinfo(i,j)  Ablockinfo[(i)*c_blocks   + (j)]
#define  Bblockinfo(i,j)  Bblockinfo[(i)*c_blocks   + (j)]
#define A(i,j) ((Ablockinfo(i,j)-1)*size_b*size_b)
#define B(i,j) ((Bblockinfo(i,j)-1)*size_b*size_b)

//============================================================

#define ldb m
#define lda m
#define ldc m


// every multiprocessor handles one BCSR-block
__global__ void 
zbcsrlupivloc_kernel( 
                       int size_b,
                       int kblocks,   
                       double **A,  
                       magma_int_t *ipiv)
{
    if( blockIdx.x < kblocks ) {
        if(threadIdx.x < size_b ){
            for( int i=0; i<size_b; i++){
                int dst = ipiv[i]-1;
                if( dst != i ){
                    double *A1 = A[blockIdx.x]+threadIdx.x*size_b+i;
                    double *A2 = A[blockIdx.x]+threadIdx.x*size_b+dst;
                    double tmp = *A2;
                    *A2 = *A1;
                    *A1 = tmp;
                }               
            }
            
        }
    }

}





/**
    Purpose
    -------
    
    For a Block-CSR ILU factorization, this routine updates all blocks in
    the trailing matrix.
    
    Arguments
    ---------

    @param
    size_b      magma_int_t
                blocksize in BCSR
    
    @param
    kblocks     magma_int_t
                number of blocks
                
    @param
    dA          magmaDoubleComplex**
                matrix in BCSR

    @param
    ipiv        magma_int_t*
                array containing pivots

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbcsrlupivloc( magma_int_t size_b, 
                    magma_int_t kblocks,
                    magmaDoubleComplex **dA,  
                    magma_int_t *ipiv ){

#if defined(PRECISION_d)
    dim3 threads( 64, 1 );

    dim3 grid(kblocks, 1, 1);
    zbcsrlupivloc_kernel<<< grid, threads, 0, magma_stream >>>( 
                  size_b, kblocks, dA, ipiv );

#endif


    return MAGMA_SUCCESS;
}



