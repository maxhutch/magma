/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/

#include "common_magma.h"
#include "magmasparse_z.h"
#include "magma.h"


#define PRECISION_z
#define BLOCKSIZE 256


__global__ void 
magma_zbajac_csr_ls_kernel(int localiters, int n, 
                            magmaDoubleComplex *valD, 
                            magma_index_t *rowD, 
                            magma_index_t *colD, 
                            magmaDoubleComplex *valR, 
                            magma_index_t *rowR,
                            magma_index_t *colR, 
                            const magmaDoubleComplex * __restrict__ b,                            
                            magmaDoubleComplex *x ){

    int ind_diag =  blockIdx.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;   


    if(index<n){
    
        start=rowR[index];
        end  =rowR[index+1];

        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex bl, tmp = zero, v = zero; 

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        start=rowD[index];
        end  =rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        /* add more local iterations */           
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - ind_diag];

            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        x[index] = local_x[threadIdx.x];
    }
}



__global__ void 
magma_zbajac_csr_kernel(    int n, 
                            magmaDoubleComplex *valD, 
                            magma_index_t *rowD, 
                            magma_index_t *colD, 
                            magmaDoubleComplex *valR, 
                            magma_index_t *rowR,
                            magma_index_t *colR, 
                            magmaDoubleComplex *b,                                
                            magmaDoubleComplex *x ){

    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int i, start, end;   

    if(index<n){
        
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex bl, tmp = zero, v = zero; 

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif

        start=rowR[index];
        end  =rowR[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        v =  bl - v;

        start=rowD[index];
        end  =rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        x[index] = x[index] + ( v - tmp ) / (valD[start]); 
    }
}









/**
    Purpose
    -------
    
    This routine is a block-asynchronous Jacobi iteration performing s
    local Jacobi-updates within the block. Input format is two CSR matrices,
    one containing the diagonal blocks, one containing the rest.

    Arguments
    ---------

    @param
    localiters  magma_int_t
                number of local Jacobi-like updates

    @param
    D           magma_z_sparse_matrix
                input matrix with diagonal blocks

    @param
    R           magma_z_sparse_matrix
                input matrix with non-diagonal parts

    @param
    b           magma_z_vector
                RHS

    @param
    x           magma_z_vector*
                iterate/solution

    

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbajac_csr(   magma_int_t localiters,
                    magma_z_sparse_matrix D,
                    magma_z_sparse_matrix R,
                    magma_z_vector b,
                    magma_z_vector *x ){

    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;

    int dimgrid1 = ( D.num_rows + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    if( R.nnz > 0 ){ 
        if( localiters == 1 )
        magma_zbajac_csr_kernel<<< grid, block, 0, magma_stream >>>
            ( D.num_rows, D.val, D.row, D.col, 
                            R.val, R.row, R.col, b.val, x->val );
        else
            magma_zbajac_csr_ls_kernel<<< grid, block, 0, magma_stream >>>
            ( localiters, D.num_rows, D.val, D.row, D.col, 
                            R.val, R.row, R.col, b.val, x->val );
    }
    else{
        printf("error: all elements in diagonal block.\n");
    }

    return MAGMA_SUCCESS;
}



