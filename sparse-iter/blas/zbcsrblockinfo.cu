/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


#define  blockinfo(i,j)  blockinfo[(i)*c_blocks   + (j)]
#define  val(i,j) val+((blockinfo(i,j)-1)*size_b*size_b)



// every thread initializes one entry
__global__ void 
zbcsrblockinfo5_kernel( 
                  magma_int_t num_blocks,
                  magmaDoubleComplex *address,
                  magmaDoubleComplex **AII ){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < num_blocks ){
        *AII[ i ] = *address;
        if(i==0)
        printf("address: %d\n", address);
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
    lustep      magma_int_t
                lustep

    @param
    num_blocks  magma_int_t
                number of nonzero blocks

    @param
    c_blocks    magma_int_t
                number of column-blocks
                
    @param
    size_b      magma_int_t
                blocksize
                
    @param
    blockinfo   magma_int_t*
                block filled? location?

    @param
    val         magmaDoubleComplex*
                pointers to the nonzero blocks in A

    @param
    AII         magmaDoubleComplex**
                pointers to the respective nonzero blocks in B


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbcsrblockinfo5(  magma_int_t lustep,
                        magma_int_t num_blocks, 
                        magma_int_t c_blocks, 
                        magma_int_t size_b,
                        magma_index_t *blockinfo,
                        magmaDoubleComplex *val,
                        magmaDoubleComplex **AII ){

 
        dim3 dimBlock( BLOCK_SIZE, 1, 1 );

        int dimgrid = (num_blocks+BLOCK_SIZE-1)/BLOCK_SIZE;
        dim3 dimGrid( dimgrid, 1, 1 );


        printf("dim grid: %d x %d", dimgrid, BLOCK_SIZE);
        magmaDoubleComplex **hAII;
        magma_malloc((void **)&hAII, num_blocks*sizeof(magmaDoubleComplex*));

        for(int i=0; i<num_blocks; i++){
           hAII[i] = val(lustep,lustep);
        }
        magma_setvector( num_blocks, sizeof(magmaDoubleComplex*), 
                                                            hAII, 1, AII, 1 );
/*
    magma_setvector( 1, sizeof(magmaDoubleComplex*), address, 1, daddress, 1 );
    zbcsrblockinfo5_kernel<<<dimGrid,dimBlock, 0, magma_stream >>>
                        ( num_blocks, daddress, AII );

*/
        return MAGMA_SUCCESS;

}



