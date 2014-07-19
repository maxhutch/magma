/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> c d s

*/

#include "common_magma.h"
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"


// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include "sm_32_intrinsics.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  64


#define PRECISION_z



// every row is handled by one threadblock
__global__ void 
magma_zaic_csr_s_kernel( magma_int_t num_rows, 
                         magma_int_t nnz,
                         const magmaDoubleComplex * __restrict__ A_val, 
                         magmaDoubleComplex *val,
                         magma_index_t *rowptr,
                         magma_index_t *rowidx, 
                         magma_index_t *colidx,
                         magmaDoubleComplex *A2 ){

    int i, j;
    int k = (blockDim.x * blockIdx.x + threadIdx.x);// % nnz;
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;

    if (k < nnz)
    {     
        i = rowidx[k];
        j = colidx[k];

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        s = __ldg( A_val+k );
#else
        s = A_val[k];
#endif

        il = rowptr[i];
        iu = rowptr[j];

        while (il < rowptr[i+1] && iu < rowptr[j+1])
        {
            sp = zero;
            jl = colidx[il];
            ju = colidx[iu];

            // avoid branching
            sp = ( jl == ju ) ? val[il] * val[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
/*
            if (jl < ju)
                il++;
            else if (ju < jl)
                iu++;
            else
            {
                // we are going to modify this u entry
                sp = val[il] * val[iu];
                s -= sp;
                il++;
                iu++;
            }
*/
        }
        // undo the last operation (it must be the last)
        s += sp;
        __syncthreads();

        // modify entry
        if (i == j)
            A2[k] = MAGMA_Z_MAKE(sqrt(abs(MAGMA_Z_REAL(s))), 0.0);
        else
            A2[k] =  s / val[rowptr[j+1]-1];
    }

}// kernel 










/**
    Purpose
    -------
    
    This routine computes the IC approximation of a matrix iteratively. 
    The idea is according to Edmond Chow's presentation at SIAM 2014.
    The input format of the matrix is Magma_CSRCOO. 

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A - initial guess (lower triangular)

    @param
    A_CSR       magma_z_sparse_matrix
                input/output matrix containing the IC approximation

    @ingroup magmasparse_zsygpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zaic_csr_s( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSR ){



    
    int blocksize1 = 1;
    int blocksize2 = 1;

    int dimgrid1 = ( A.nnz + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    magma_z_vector A2;


            // init DEV vectors
    magma_z_vinit( &A2, Magma_DEV, A.nnz, MAGMA_Z_ONE );    

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    magma_zaic_csr_s_kernel<<< grid, block, 0, magma_stream >>>
            ( A.num_rows, A.nnz,  A.val, A_CSR.val, A_CSR.row, 
                                    A_CSR.rowidx,  A_CSR.col, A2.val );

    magma_zcopy( A.nnz, A2.val, 1, A_CSR.val, 1 );                            // rr = b

    magma_z_vfree(&A2);

    return MAGMA_SUCCESS;
}



