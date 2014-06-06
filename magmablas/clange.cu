/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:45 2013
       @author Mark Gates
*/
#include "common_magma.h"

/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:m-1, for || A ||_inf,
 * where m and n are any size.
 * Has ceil( m/64 ) blocks of 64 threads. Each thread does one row. */
extern "C" __global__ void
clange_inf_kernel(
    int m, int n, const magmaFloatComplex *A, int lda, float *dwork )
{
    int i = blockIdx.x*64 + threadIdx.x;
    float Cb[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            const magmaFloatComplex *Aend = A + lda*n;
            magmaFloatComplex rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                Cb[0] += cuCabsf( rA[0] );  rA[0] = A[0];
                Cb[1] += cuCabsf( rA[1] );  rA[1] = A[lda];
                Cb[2] += cuCabsf( rA[2] );  rA[2] = A[2*lda];
                Cb[3] += cuCabsf( rA[3] );  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            Cb[0] += cuCabsf( rA[0] );
            Cb[1] += cuCabsf( rA[1] );
            Cb[2] += cuCabsf( rA[2] );
            Cb[3] += cuCabsf( rA[3] );
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                Cb[0] += cuCabsf( A[0] );
                break;
    
            case 2:
                Cb[0] += cuCabsf( A[0]   );
                Cb[1] += cuCabsf( A[lda] );
                break;
    
            case 3:
                Cb[0] += cuCabsf( A[0]     );
                Cb[1] += cuCabsf( A[lda]   );
                Cb[2] += cuCabsf( A[2*lda] );
                break;
        }
    
        /* compute final result */
        dwork[i] = Cb[0] + Cb[1] + Cb[2] + Cb[3];
    }
}

extern "C" float
magmablas_clange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    const magmaFloatComplex *A, magma_int_t lda, float *dwork )
{
/*
    Purpose
    =======
    CLANGE  returns the value of the one norm, or the Frobenius norm, or
    the  infinity norm, or the  element of  largest absolute value  of a
    real matrix A.
    
    Description
    ===========
    CLANGE returns the value
    
       CLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'            ** not yet supported
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'       ** not yet supported
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'  ** not yet supported
    
    where  norm1  denotes the  one norm of a matrix (maximum column sum),
    normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
    normF  denotes the  Frobenius norm of a matrix (square root of sum of
    squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm.
    
    Arguments
    =========
    NORM    (input) CHARACTER*1
            Specifies the value to be returned in CLANGE as described
            above.
    
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            CLANGE is set to zero.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            CLANGE is set to zero.
    
    A       (input) REAL array on the GPU, dimension (LDA,N)
            The m by n matrix A.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(M,1).
    
    DWORK   (workspace) REAL array on the GPU, dimension (MAX(1,LWORK)),
            where LWORK >= M when NORM = 'I'; otherwise, WORK is not
            referenced.
    ====================================================================  */
    
    magma_int_t info = 0;
    if ( norm != 'I' && norm != 'i' )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( lda < m )
        info = -5;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( m == 0 || n == 0 )
        return 0;
    
    dim3 threads( 64 );
    dim3 grid( (m-1)/64 + 1 );
    clange_inf_kernel<<< grid, threads, 0, magma_stream >>>( m, n, A, lda, dwork );
    int i = cublasIsamax( m, dwork, 1 ) - 1;
    float res;
    cudaMemcpy( &res, &dwork[i], sizeof(float), cudaMemcpyDeviceToHost );
    return res;
}
