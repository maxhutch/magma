/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from zbajac_csr.cu normal z -> d, Sat Nov 15 19:54:21 2014

*/

#include "common_magma.h"
#include "magmasparse_d.h"
#include "magma.h"


#define PRECISION_d
#define BLOCKSIZE 256


__global__ void
magma_dbajac_csr_ls_kernel(int localiters, int n, 
                            magmaDouble_ptr valD, 
                            magmaIndex_ptr rowD, 
                            magmaIndex_ptr colD, 
                            magmaDouble_ptr valR, 
                            magmaIndex_ptr rowR,
                            magmaIndex_ptr colR, 
                            const magmaDouble_ptr  __restrict__ b,                            
                            magmaDouble_ptr x ){

    int inddiag =  blockIdx.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;   


    if(index<n){
    
        start=rowR[index];
        end  =rowR[index+1];

        double zero = MAGMA_D_MAKE(0.0, 0.0);
        double bl, tmp = zero, v = zero; 

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
        __shared__ double local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        x[index] = local_x[threadIdx.x];
    }
}



__global__ void
magma_dbajac_csr_kernel(    
    int n, 
    magmaDouble_ptr valD, 
    magmaIndex_ptr rowD, 
    magmaIndex_ptr colD, 
    magmaDouble_ptr valR, 
    magmaIndex_ptr rowR,
    magmaIndex_ptr colR, 
    magmaDouble_ptr b,                                
    magmaDouble_ptr x ){

    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int i, start, end;   

    if(index<n){
        
        double zero = MAGMA_D_MAKE(0.0, 0.0);
        double bl, tmp = zero, v = zero; 

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

    @param[in]
    localiters  magma_int_t
                number of local Jacobi-like updates

    @param[in]
    D           magma_d_sparse_matrix
                input matrix with diagonal blocks

    @param[in]
    R           magma_d_sparse_matrix
                input matrix with non-diagonal parts

    @param[in]
    b           magma_d_vector
                RHS

    @param[in]
    x           magma_d_vector*
                iterate/solution

    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbajac_csr(
    magma_int_t localiters,
    magma_d_sparse_matrix D,
    magma_d_sparse_matrix R,
    magma_d_vector b,
    magma_d_vector *x,
    magma_queue_t queue )
{
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;

    int dimgrid1 = ( D.num_rows + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    if ( R.nnz > 0 ) { 
        if ( localiters == 1 )
        magma_dbajac_csr_kernel<<< grid, block, 0, queue >>>
            ( D.num_rows, D.dval, D.drow, D.dcol, 
                            R.dval, R.drow, R.dcol, b.dval, x->dval );
        else
            magma_dbajac_csr_ls_kernel<<< grid, block, 0, queue >>>
            ( localiters, D.num_rows, D.dval, D.drow, D.dcol, 
                            R.dval, R.drow, R.dcol, b.dval, x->dval );
    }
    else {
        printf("error: all elements in diagonal block.\n");
    }

    return MAGMA_SUCCESS;
}



