/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlascl_diag.cu normal z -> d, Fri Jan 30 19:00:09 2015
*/
#include "common_magma.h"

#define NB 64


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
dlascl_diag_lower(int m, int n, magmaDouble_const_ptr D, int ldd, 
                                      magmaDouble_ptr A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;
    if (ind < m) {
        for(int j=0; j < n; j++ )
            A[j*lda] /= D[j+j*ldd];
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
dlascl_diag_upper(int m, int n, magmaDouble_const_ptr D, int ldd, 
                                      magmaDouble_ptr A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;
    if (ind < m) {
        for(int j=0; j < n; j++ )
            A[j*lda] /= D[ind+ind*ldd];
    }
}


/**
    Purpose
    -------
    DLASCL2 scales the M by N real matrix A by the real diagonal matrix dD.
    TYPE specifies that A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    \param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    \param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    \param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    \param[in]
    dD      DOUBLE PRECISION vector, dimension (M)
            The diagonal matrix containing the scalar factors. Stored as a vector.

    \param[in,out]
    dA      DOUBLE PRECISION array, dimension (LDDA,N)
            The matrix to be scaled by dD.  See TYPE for the
            storage type.

    \param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    \param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl_diag_q(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD, magma_int_t lddd, 
          magmaDouble_ptr dA, magma_int_t ldda, 
    magma_int_t *info, magma_queue_t queue )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( n < 0 )
        *info = -3;
    else if ( ldda < max(1,m) )
        *info = -5;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 grid( (m + NB - 1)/NB );
    dim3 threads( NB );
    
    if (type == MagmaLower) {
        dlascl_diag_lower <<< grid, threads, 0, queue >>> (m, n, dD, lddd, dA, ldda);
    }
    else if (type == MagmaUpper) {
        dlascl_diag_upper <<< grid, threads, 0, queue >>> (m, n, dD, lddd, dA, ldda);
    }
}


/**
    @see magmablas_dlascl2_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD, magma_int_t lddd, 
          magmaDouble_ptr dA, magma_int_t ldda, 
    magma_int_t *info )
{
    magmablas_dlascl_diag_q( type, m, n, dD, lddd, dA, ldda, info, magma_stream );
}
