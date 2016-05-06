/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar

       @generated from src/zposv_batched.cpp normal z -> s, Mon May  2 23:30:28 2016
*/
#include "magma_internal.h"
/**
    Purpose
    -------
    SPOSV computes the solution to a real system of linear equations
       A * X = B,
    where A is an N-by-N symmetric positive definite matrix and X and B
    are N-by-NRHS matrices.
    The Cholesky decomposition is used to factor A as
       A = U**H * U,  if UPLO = MagmaUpper, or
       A = L * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and  L is a lower triangular
    matrix.  The factored form of A is then used to solve the system of
    equations A * X = B.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
             Each is a REAL array on the GPU, dimension (LDDA,N)
             On entry, each pointer is a symmetric matrix A.  
             If UPLO = MagmaUpper, the leading
             N-by-N upper triangular part of A contains the upper
             triangular part of the matrix A, and the strictly lower
             triangular part of dA is not referenced.  If UPLO = MagmaLower, the
             leading N-by-N lower triangular part of A contains the lower
             triangular part of the matrix A, and the strictly upper
             triangular part of A is not referenced.
    \n
             On exit, if corresponding entry in dinfo_array = 0, 
             each pointer is the factor U or L from the Cholesky
             factorization A = U**H*U or A = L*L**H.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,N).

    @param[in,out]
    dB_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDB,NRHS)
            On entry, each pointer is a right hand side matrix B.
            On exit, each pointer is the corresponding solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  LDB >= max(1,N).

    @param[out]
    dinfo_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sposv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_sposv_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  float **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount, magma_queue_t queue)
{
    /* Local variables */
    magma_int_t info = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    if ( n < 0 )
        info = -2;
    if ( nrhs < 0 )
        info = -3;
    if ( ldda < max(1, n) )
        info = -5;
    if ( lddb < max(1, n) )
        info = -7;
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return info;
    }

    info = magma_spotrf_batched( uplo, n, dA_array, ldda, dinfo_array, batchCount, queue);
    if ( info != MAGMA_SUCCESS ) {
        return info;
    }


#ifdef CHECK_INFO
    // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
    magma_int_t *cpu_info = NULL;
    magma_imalloc_cpu( &cpu_info, batchCount );
    magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
    for (magma_int_t i=0; i < batchCount; i++)
    {
        if (cpu_info[i] != 0 ) {
            printf("magma_spotrf_batched matrix %d returned error %d\n",i, (int)cpu_info[i] );
            info = cpu_info[i];
            magma_free_cpu (cpu_info);
            return info;
        }
    }
    magma_free_cpu (cpu_info);
#endif

    info = magma_spotrs_batched( uplo, n, nrhs, dA_array, ldda, dB_array, lddb,  batchCount, queue );
    return info;
}
