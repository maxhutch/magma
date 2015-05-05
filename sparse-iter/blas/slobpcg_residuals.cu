/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zlobpcg_residuals.cu normal z -> s, Sun May  3 11:22:58 2015

*/

#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512
#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

#define PRECISION_s


// copied from snrm2.cu in trunk/magmablas
// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
template< int n >
__device__ void sum_reduce( /*int n,*/ int i, magmaFloat_ptr  x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce



__global__ void
magma_slobpcg_res_kernel( 
    magma_int_t num_rows, 
    magma_int_t num_vecs, 
    magmaFloat_ptr evals, 
    float * X, 
    float * R,
    magmaFloat_ptr res)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x; // global row index

    if( row<num_rows){
        for( int i=0; i<num_vecs; i++ ){ 

            R[row + i*num_rows] = R[row + i*num_rows] 
                                    + MAGMA_S_MAKE( -evals[i], 0.0 )
                                                * X[ row + i*num_rows ];
        }
    }
}


/*
magmablas_snrm2_kernel( 
    int m, 
    float * da, 
    int ldda, 
    float * dxnorm )
{
    const int i = threadIdx.x;
    magmaFloat_ptr dx = da + blockIdx.x * ldda;

    __shared__ float sum[ BLOCK_SIZE ];
    float re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {

#if (defined(PRECISION_s) || defined(PRECISION_d))
        re = dx[j];
        lsum += re*re;
#else
        re = MAGMA_S_REAL( dx[j] );
        float im = MAGMA_S_IMAG( dx[j] );
        lsum += re*re + im*im;
#endif

    }
    sum[i] = lsum;
    sum_reduce< BLOCK_SIZE >( i, sum );
    
    if (i==0)
        res[blockIdx.x] = sqrt(sum[0]);
}
*/



/**
    Purpose
    -------
    
    This routine computes for Block-LOBPCG, the set of residuals. 
                            R = Ax - x evalues
    It replaces:
    for(int i=0; i < n; i++){
        magma_saxpy(m, MAGMA_S_MAKE(-evalues[i],0),blockX+i*m,1,blockR+i*m,1);
    }
    The memory layout of x is:

        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    x = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[0] x2[1] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /
    
    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows

    @param[in]
    num_vecs    magma_int_t
                number of vectors
                
    @param[in]
    evalues     magmaFloat_ptr 
                array of eigenvalues/approximations

    @param[in]
    X           magmaFloat_ptr 
                block of eigenvector approximations
                
    @param[in]
    R           magmaFloat_ptr 
                block of residuals

    @param[in]
    res         magmaFloat_ptr 
                array of residuals

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_slobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaFloat_ptr evalues, 
    magmaFloat_ptr X,
    magmaFloat_ptr R, 
    magmaFloat_ptr res,
    magma_queue_t queue )
{
    // every thread handles one row

    magma_int_t block_size = BLOCK_SIZE;
 
    dim3 threads( block_size );
    dim3 grid( magma_ceildiv( num_rows, block_size ) );

    magma_slobpcg_res_kernel<<< grid, threads, 0, queue >>>
                                ( num_rows, num_vecs, evalues, X, R, res );


    return MAGMA_SUCCESS;
}



