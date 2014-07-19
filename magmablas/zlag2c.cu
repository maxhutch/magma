/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions mixed zc -> ds
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_z
#define blksize 64

// TODO get rid of global variable!
static __device__ int flag = 0; 

__global__ void 
zlag2c_kernel( int m, int n, 
               const magmaDoubleComplex *A, int lda, 
               magmaFloatComplex *SA,       int ldsa, 
               double rmax ) 
{
    magmaDoubleComplex tmp;
    double neg_rmax = - rmax;
    
    int i = blockIdx.x*blksize + threadIdx.x;
    if ( i < m ) {
        A  += i;
        SA += i; 
        const magmaDoubleComplex *Aend = A + lda*n;
        while( A < Aend ) {
            tmp = *A;
            if (   (cuCreal(tmp) < neg_rmax) || (cuCreal(tmp) > rmax)
#if defined(PRECISION_z) || defined(PRECISION_c)
                || (cuCimag(tmp) < neg_rmax) || (cuCimag(tmp) > rmax) 
#endif
                )
            {
                flag = 1; 
            }
            *SA = cuComplexDoubleToFloat( tmp );
            A  += lda;
            SA += ldsa;
        }
    }
}


/**
    Purpose
    -------
    ZLAG2C converts a double-complex matrix, A,
                 to a single-complex matrix, SA.
    
    RMAX is the overflow for the single-complex arithmetic.
    ZLAG2C checks that all the entries of A are between -RMAX and
    RMAX. If not, the conversion is aborted and a flag is raised.
        
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of lines of the matrix A.  m >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       COMPLEX_16 array, dimension (LDA,n)
            On entry, the m-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,m).
    
    @param[out]
    SA      COMPLEX array, dimension (LDSA,n)
            On exit, if INFO=0, the m-by-n coefficient matrix SA; if
            INFO>0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,m).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void 
magmablas_zlag2c( magma_int_t m, magma_int_t n, 
                  const magmaDoubleComplex *A, magma_int_t lda, 
                  magmaFloatComplex *SA,       magma_int_t ldsa, 
                  magma_int_t *info ) 
{
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldsa < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    double rmax = (double)lapackf77_slamch("O");

    dim3 threads( blksize );
    dim3 grid( (m+blksize-1)/blksize );
    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    zlag2c_kernel<<< grid, threads, 0, magma_stream >>>( m, n, A, lda, SA, ldsa, rmax ); 
    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
}
