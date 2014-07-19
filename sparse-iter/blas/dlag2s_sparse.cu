/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zlag2c_sparse.cu mixed zc -> ds, Fri Jul 18 17:34:27 2014

*/
#include "common_magma.h"
#include "../include/magmasparse_z.h"
#include "../include/magmasparse_ds.h"
#include "../../include/magma.h"
#include "../include/mmio.h"
#include "common_magma.h"

#define PRECISION_d
#define BLOCKSIZE 512

#define min(a, b) ((a) < (b) ? (a) : (b))

// TODO get rid of global variable!
__device__ int flag = 0; 

__global__ void 
magmaint_dlag2s_sparse(  int M, int N, 
                  const double *A,
                  float *SA ){

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; 
                                    // global thread index

    if( thread_id < M ){
        for( int i=0; i<N; i++ ){

            SA[i*M+thread_id] = (float)( A[i*M+thread_id] );

        }
    } 
}


/** 
    Purpose
    -------
    DLAG2S converts a DOUBLE PRECISION matrix A to a SINGLE PRECISION
    matrix SA.
    
    RMAX is the overflow for the SINGLE PRECISION arithmetic.
    DLAG2S checks that all the entries of A are between -RMAX and
    RMAX. If not the convertion is aborted and a flag is raised.
        
    Arguments
    ---------
    @param[in]
    M       INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    @param[in]
    N       INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    @param[in]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    @param[out]
    SA      SINGLE PRECISION array, dimension (LDSA,N)
            On exit, if INFO=0, the M-by-N coefficient matrix SA; if
            INFO>0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the SINGLE PRECISION
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.

    @ingroup magmasparse_zaux
    ********************************************************************/
extern "C" void
magmablas_dlag2s_sparse( magma_int_t M, magma_int_t N, 
                  const double *A, magma_int_t lda, 
                  float *SA,       magma_int_t ldsa, 
                  magma_int_t *info ) 
{
    /*
    (TODO note from original dense source)
    
    Note
    ----
          - We have to provide INFO at the end that dlag2s isn't doable now. 
          - Transfer a single value TO/FROM CPU/GPU
          - SLAMCH that's needed is called from underlying BLAS
          - Only used in iterative refinement
          - Do we want to provide this in the release?
    */
    
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


    dim3 grid( (M+BLOCKSIZE-1)/BLOCKSIZE, 1, 1);


    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    magmaint_dlag2s_sparse<<< grid, BLOCKSIZE, 0, magma_stream >>>
                                        ( M, N, A, SA ) ; 
    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
    
}



__global__ void 
magma_dlag2s_CSR_DENSE_kernel( int num_rows, int num_cols, 
                               double *Aval, magma_index_t *Arow, 
                               magma_index_t *Acol, float *Bval ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        for( j=0; j<num_cols; j++)
            Bval[ j ] = MAGMA_S_MAKE(0.0, 0.0);
        int start = Arow[ row ];
        int end = Arow[ row+1 ];
        for( j=start; j<end; j++ )
            Bval[ row*num_rows+Acol[j] ] = (float)( Aval[ j] );
    }
}

__global__ void 
magma_dlag2s_CSR_DENSE_kernel_1( int num_rows, int num_cols, 
                                 float *Bval ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        for( j=0; j<num_cols; j++)
            Bval[ j ] = MAGMA_S_MAKE(0.0, 0.0);
    }
}

__global__ void 
magma_dlag2s_CSR_DENSE_kernel_2( int num_rows, int num_cols, 
                               double *Aval, magma_index_t *Arow, 
                               magma_index_t *Acol, float *Bval ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        int start = Arow[ row ];
        int end = Arow[ row+1 ];
        for( j=start; j<end; j++ )
            Bval[ row*num_rows+Acol[j] ] = (float)( Aval[ j] );
    }
}




extern "C" void 
magma_dlag2s_CSR_DENSE(       magma_d_sparse_matrix A, 
                              magma_s_sparse_matrix *B ){

    magma_int_t stat;

    if( A.memory_location == Magma_DEV && A.storage_type == Magma_CSR){
        B->storage_type = Magma_DENSE;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        stat = magma_smalloc( &B->val, A.num_rows* A.num_cols );
        if( stat != 0 ) 
        {printf("Memory Allocation Error converting matrix\n"); exit(0); }
        
        dim3 Bs( BLOCKSIZE );
        dim3 Gs( (A.num_rows+BLOCKSIZE-1)/BLOCKSIZE );

        magma_dlag2s_CSR_DENSE_kernel<<< Bs, Gs, 0, magma_stream >>>
        ( A.num_rows, A.num_cols, A.val, A.row, A.col, B->val );
    }
}



extern "C" void 
magma_dlag2s_CSR_DENSE_alloc( magma_d_sparse_matrix A, 
                              magma_s_sparse_matrix *B ){

    magma_int_t stat;

    if( A.memory_location == Magma_DEV && A.storage_type == Magma_CSR){
        B->storage_type = Magma_DENSE;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        stat = magma_smalloc( &B->val, A.num_rows* A.num_cols );
        if( stat != 0 ) 
        {printf("Memory Allocation Error converting matrix\n"); exit(0); }
        
        dim3 Bs( BLOCKSIZE );
        dim3 Gs( (A.num_rows+BLOCKSIZE-1)/BLOCKSIZE );

        magma_dlag2s_CSR_DENSE_kernel_1<<< Bs, Gs, 0, magma_stream >>>
        ( A.num_rows, A.num_cols, B->val );
    }
}


extern "C" void 
magma_dlag2s_CSR_DENSE_convert( magma_d_sparse_matrix A, 
                                magma_s_sparse_matrix *B ){


    if( B->memory_location == Magma_DEV && B->storage_type == Magma_DENSE){
        dim3 Bs( BLOCKSIZE );
        dim3 Gs( (A.num_rows+BLOCKSIZE-1)/BLOCKSIZE );

        magma_dlag2s_CSR_DENSE_kernel_2<<< Bs, Gs, 0, magma_stream >>>
        ( A.num_rows, A.num_cols, A.val, A.row, A.col, B->val );
    }
}
