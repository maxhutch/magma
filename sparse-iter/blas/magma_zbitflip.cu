/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define PRECISION_z


__global__ void 
magma_zbitflip_kernel( magmaDoubleComplex *d, int loc, int bit ){
    int z = blockDim.x * blockIdx.x + threadIdx.x ;

    if (z == 0 ){
#if ( defined(PRECISION_d) )
        d[loc] = ((long int) d[loc]) ^ ((long int)1 << bit ) ;
#elif ( defined(PRECISION_s) )
        d[loc] = ((int) d[loc]) ^ ((int)1 << bit ) ;
#endif
    }
}




/** -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    -------

    This tool flips the bit 'bit' of the magmaDoubleComplex value d[ loc ]. 

    @ingroup magmasparse_zaux
    ********************************************************************/




extern "C" magma_int_t
magma_zbitflip( magmaDoubleComplex *d, magma_int_t loc, magma_int_t bit ){


    int blocksize1 = 1;
    int blocksize2 = 1;

    int dimgrid1 = 1;
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
   
    magma_zbitflip_kernel<<< grid, block, 0, magma_stream >>>
        ( d, loc, bit );
       
    return MAGMA_SUCCESS; 
}



