/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zbcsrcpy.cu normal z -> d, Fri Jul 18 17:34:27 2014

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



// every multiprocessor handles one BCSR-block to copy from A
__global__ void 
dbcsrvalcpy_kernel( 
                  int size_b,
                  magma_int_t num_blocks,
                  double **Aval, 
                  double **Bval ){
    if(blockIdx.x*65535+blockIdx.y < num_blocks){
        double *dA = Aval[ blockIdx.x*65535+blockIdx.y ];
        double *dB = Bval[ blockIdx.x*65535+blockIdx.y ];
        int i = threadIdx.x;

        while( i<size_b*size_b ){
                dB[i] = dA[i];
                i+=BLOCK_SIZE;
        }
    }
}

// every multiprocessor handles one BCSR-block to initialize with 0
__global__ void 
dbcsrvalzro_kernel( 
                  int size_b,
                  magma_int_t num_blocks,
                  double **Bval ){
    if(blockIdx.x*65535+blockIdx.y < num_blocks){
        double *dB = Bval[ blockIdx.x*65535+blockIdx.y ];
        int i = threadIdx.x;
        //dB += i;

        while( i<size_b*size_b ){
                dB[i] = MAGMA_D_MAKE(0.0, 0.0);
                i+=BLOCK_SIZE;
        }
    }

}



/**
    Purpose
    -------
    
    For a Block-CSR ILU factorization, this routine copies the filled blocks
    from the original matrix A and initializes the blocks that will later be 
    filled in the factorization process with zeros.
    
    Arguments
    ---------


    @param
    size_b      magma_int_t
                blocksize in BCSR

    @param
    num_blocks  magma_int_t
                number of nonzero blocks

    @param
    num_zblocks magma_int_t
                number of zero-blocks (will later be filled)

    @param
    Aval        double**
                pointers to the nonzero blocks in A

    @param
    Bval        double**
                pointers to the nonzero blocks in B

    @param
    Bval2        double**
                pointers to the zero blocks in B


    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbcsrvalcpy(  magma_int_t size_b, 
                    magma_int_t num_blocks, 
                    magma_int_t num_zblocks, 
                    double **Aval, 
                    double **Bval,
                    double **Bval2 ){

 
        dim3 dimBlock( BLOCK_SIZE, 1, 1 );

        // the grids are adapted to the number of nonzero/zero blocks 
        // the upper block-number the kernels can handle is 65535*65535
        int dimgrid1 = 65535;
        int dimgrid2 = (num_blocks+65535-1)/65535;
        int dimgrid3 = (num_zblocks+65535-1)/65535;
        dim3 dimGrid( dimgrid2, dimgrid1, 1 );

        dbcsrvalcpy_kernel<<<dimGrid,dimBlock, 0, magma_stream >>>
                            ( size_b, num_blocks, Aval, Bval );

        dim3 dimGrid2( dimgrid3, dimgrid1, 1 );

        dbcsrvalzro_kernel<<<dimGrid2,dimBlock, 0, magma_stream >>>
                            ( size_b, num_zblocks, Bval2 );

        return MAGMA_SUCCESS;

}



