/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from clag2z_sparse.cu mixed zc -> ds, Fri May 30 10:41:34 2014

*/
#include "common_magma.h"
#include "../include/magmasparse_z.h"
#include "../include/magmasparse_ds.h"
#include "../../include/magma.h"
#include "../include/mmio.h"
#include "common_magma.h"

#define PRECISION_d
#define blksize 512

#define min(a, b) ((a) < (b) ? (a) : (b))

// TODO get rid of global variable!
__device__ int flag = 0; 

__global__ void 
magmaint_slag2d_sparse(  int M, int N, 
                  const float *SA, int ldsa, 
                  double *A,       int lda, 
                  double RMAX ) 
{
    int inner_bsize = blockDim.x;
    int outer_bsize = inner_bsize * 512;
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; 
            // global thread index

    if( thread_id < M ){
        for( int i= outer_bsize * blockIdx.x  + threadIdx.x ; 
            i<min( M, outer_bsize * ( blockIdx.x + 1));  i+=inner_bsize){
            A[i] = (double)( SA[i] );

        }
    } 
}

/*
    Note
    ====
          - We have to provide INFO at the end that dlag2s isn't doable now. 
          - Transfer a single value TO/FROM CPU/GPU
          - SLAMCH that's needed is called from underlying BLAS
          - Only used in iterative refinement
          - Do we want to provide this in the release?
    
    Purpose
    =======
    SLAG2D converts a SINGLE PRECISION matrix SA to a DOUBLE PRECISION
    matrix A.
    
    RMAX is the overflow for the SINGLE PRECISION arithmetic.
    SLAG2D checks that all the entries of A are between -RMAX and
    RMAX. If not the convertion is aborted and a flag is raised.
        
    Arguments
    =========
    M       (input) INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    SA      (input) SINGLE PRECISION array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.
    
    LDSA    (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    A       (output) DOUBLE PRECISION array, dimension (LDA,N)
            On exit, if INFO=0, the M-by-N coefficient matrix A; if
            INFO>0, the content of A is unspecified.
    
    LDA     (input) INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value
            = 1:  an entry of the matrix A is greater than the SINGLE PRECISION
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.
    ======================================================================    */

extern "C" void 
magmablas_slag2d_sparse( magma_int_t M, magma_int_t N , 
                  const float *SA, magma_int_t ldsa, 
                  double *A,       magma_int_t lda, 
                  magma_int_t *info ){    


    *info = 0;
    if ( M < 0 )
        *info = -1;
    else if ( N < 0 )
        *info = -2;
    else if ( lda < max(1,M) )
        *info = -4;
    else if ( ldsa < max(1,M) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }
    
    double RMAX = (double)lapackf77_slamch("O");

    int block;
    dim3 dimBlock(blksize);// Number of Threads per Block
    block = (M/blksize)/blksize;
    if(block*blksize*blksize<(M))block++;
    dim3 dimGrid(block);// Number of Blocks
   

    dim3 threads( blksize, 1, 1 );
    dim3 grid( (M+blksize-1)/blksize, 1, 1);
    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    magmaint_slag2d_sparse<<< dimGrid , dimBlock, 0, magma_stream >>>
                                        ( M, N, SA, lda, A, ldsa, RMAX ) ; 
    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
}
