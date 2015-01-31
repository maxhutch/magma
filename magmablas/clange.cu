/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlange.cu normal z -> c, Fri Jan 30 19:00:09 2015
       @author Mark Gates
*/
#include "common_magma.h"
#include "magma_templates.h"


// ----------------------------------------
// max reduction, for arbitrary size vector. Leaves max(x) in x[0].
// Uses only one thread block of 512 threads, so is not efficient for really large vectors.
template< typename T >
__global__ void
max_nan_kernel( int n, T* x )
{
    __shared__ T smax[ 512 ];
    int tx = threadIdx.x;
    
    smax[tx] = 0;
    for( int i=tx; i < n; i += 512 ) {
        smax[tx] = max_nan( smax[tx], x[i] );
    }
    magma_max_nan_reduce< 512 >( tx, smax );
    if ( tx == 0 ) {
        x[0] = smax[0];
    }
}


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:m-1, for || A ||_inf,
 * where m and n are any size.
 * Has ceil( m/64 ) blocks of 64 threads. Each thread does one row.
 * See also clange_max_kernel code, below. */
extern "C" __global__ void
clange_inf_kernel(
    int m, int n, const magmaFloatComplex *A, int lda, float *dwork )
{
    int i = blockIdx.x*64 + threadIdx.x;
    float rsum[4] = {0, 0, 0, 0};
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
                rsum[0] += cuCabsf( rA[0] );  rA[0] = A[0];
                rsum[1] += cuCabsf( rA[1] );  rA[1] = A[lda];
                rsum[2] += cuCabsf( rA[2] );  rA[2] = A[2*lda];
                rsum[3] += cuCabsf( rA[3] );  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rsum[0] += cuCabsf( rA[0] );
            rsum[1] += cuCabsf( rA[1] );
            rsum[2] += cuCabsf( rA[2] );
            rsum[3] += cuCabsf( rA[3] );
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rsum[0] += cuCabsf( A[0] );
                break;
    
            case 2:
                rsum[0] += cuCabsf( A[0]   );
                rsum[1] += cuCabsf( A[lda] );
                break;
    
            case 3:
                rsum[0] += cuCabsf( A[0]     );
                rsum[1] += cuCabsf( A[lda]   );
                rsum[2] += cuCabsf( A[2*lda] );
                break;
        }
    
        /* compute final result */
        dwork[i] = rsum[0] + rsum[1] + rsum[2] + rsum[3];
    }
}


/* Computes max of row dwork[i] = max( abs( A(i,:) )), i=0:m-1, for || A ||_max,
 * where m and n are any size.
 * Has ceil( m/64 ) blocks of 64 threads. Each thread does one row.
 * Based on clange_inf_kernel code, above. */
extern "C" __global__ void
clange_max_kernel(
    int m, int n, const magmaFloatComplex *A, int lda, float *dwork )
{
    int i = blockIdx.x*64 + threadIdx.x;
    float rmax[4] = {0, 0, 0, 0};
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
                rmax[0] = max_nan( rmax[0], cuCabsf( rA[0] ));  rA[0] = A[0];
                rmax[1] = max_nan( rmax[1], cuCabsf( rA[1] ));  rA[1] = A[lda];
                rmax[2] = max_nan( rmax[2], cuCabsf( rA[2] ));  rA[2] = A[2*lda];
                rmax[3] = max_nan( rmax[3], cuCabsf( rA[3] ));  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rmax[0] = max_nan( rmax[0], cuCabsf( rA[0] ));
            rmax[1] = max_nan( rmax[1], cuCabsf( rA[1] ));
            rmax[2] = max_nan( rmax[2], cuCabsf( rA[2] ));
            rmax[3] = max_nan( rmax[3], cuCabsf( rA[3] ));
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rmax[0] = max_nan( rmax[0], cuCabsf( A[0] ));
                break;                          
                                                
            case 2:                             
                rmax[0] = max_nan( rmax[0], cuCabsf( A[  0] ));
                rmax[1] = max_nan( rmax[1], cuCabsf( A[lda] ));
                break;                          
                                                
            case 3:                             
                rmax[0] = max_nan( rmax[0], cuCabsf( A[    0] ));
                rmax[1] = max_nan( rmax[1], cuCabsf( A[  lda] ));
                rmax[2] = max_nan( rmax[2], cuCabsf( A[2*lda] ));
                break;
        }
    
        /* compute final result */
        dwork[i] = max_nan( max_nan( max_nan( rmax[0], rmax[1] ), rmax[2] ), rmax[3] );
    }
}


/* Computes col sums dwork[j] = sum( abs( A(:,j) )), j=0:n-1, for || A ||_one,
 * where m and n are any size.
 * Has n blocks of NB threads each. Block j sums one column, A(:,j) into dwork[j].
 * Thread i accumulates A(i,j) + A(i+NB,j) + A(i+2*NB,j) + ... into ssum[i],
 * then threads collectively do a sum-reduction of ssum,
 * and finally thread 0 saves to dwork[j]. */
extern "C" __global__ void
clange_one_kernel(
    int m, int n, const magmaFloatComplex *A, int lda, float *dwork )
{
    __shared__ float ssum[64];
    int tx = threadIdx.x;
    
    A += blockIdx.x*lda;  // column j
    
    ssum[tx] = 0;
    for( int i = tx; i < m; i += 64 ) {
        ssum[tx] += cuCabsf( A[i] );
    }
    magma_sum_reduce< 64 >( tx, ssum );
    if ( tx == 0 ) {
        dwork[ blockIdx.x ] = ssum[0];
    }
}


/**
    Purpose
    -------
    CLANGE  returns the value of the one norm, or the Frobenius norm, or
    the  infinity norm, or the  element of  largest absolute value  of a
    real matrix A.
    
    Description
    -----------
    CLANGE returns the value
    
       CLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'  ** not yet supported
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of
    squares). Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    Arguments
    ---------
    @param[in]
    norm    CHARACTER*1
            Specifies the value to be returned in CLANGE as described
            above.
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            CLANGE is set to zero.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            CLANGE is set to zero.
    
    @param[in]
    A       REAL array on the GPU, dimension (LDA,N)
            The m by n matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(M,1).
    
    @param
    dwork   (workspace) REAL array on the GPU, dimension (LWORK).
    
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If NORM = 'I' or 'M', LWORK >= max( 1, M ).
            If NORM = '1',        LWORK >= max( 1, N ).
            Note this is different than LAPACK, which requires WORK only for
            NORM = 'I', and does not pass LWORK.

    @ingroup magma_caux2
    ********************************************************************/
extern "C" float
magmablas_clange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork )  //, magma_int_t lwork )
{
    magma_int_t info = 0;
    if ( ! (norm == MagmaInfNorm || norm == MagmaMaxNorm || norm == MagmaOneNorm) )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -5;
    //else if ( ((norm == MagmaInfNorm || norm == MagmaMaxNorm) && (lwork < m)) ||
    //          ((norm == MagmaOneNorm) && (lwork < n)) )
    //    info = -7;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( m == 0 || n == 0 )
        return 0;
    
    //int i;
    dim3 threads( 64 );
    float result = -1;
    if ( norm == MagmaInfNorm ) {
        dim3 grid( (m-1)/64 + 1 );
        clange_inf_kernel<<< grid, threads, 0, magma_stream >>>( m, n, dA, ldda, dwork );
        max_nan_kernel<<< 1, 512, 0, magma_stream >>>( m, dwork );
        cudaMemcpy( &result, &dwork[0], sizeof(float), cudaMemcpyDeviceToHost );
    }
    else if ( norm == MagmaMaxNorm ) {
        dim3 grid( (m-1)/64 + 1 );
        clange_max_kernel<<< grid, threads, 0, magma_stream >>>( m, n, dA, ldda, dwork );
        max_nan_kernel<<< 1, 512, 0, magma_stream >>>( m, dwork );
        cudaMemcpy( &result, &dwork[0], sizeof(float), cudaMemcpyDeviceToHost );
    }
    else if ( norm == MagmaOneNorm ) {
        dim3 grid( n );
        clange_one_kernel<<< grid, threads, 0, magma_stream >>>( m, n, dA, ldda, dwork );
        max_nan_kernel<<< 1, 512, 0, magma_stream >>>( n, dwork );
        cudaMemcpy( &result, &dwork[0], sizeof(float), cudaMemcpyDeviceToHost );
    }
    
    return result;
}
