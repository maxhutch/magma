/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Ichitaro Yamazaki
       @author Stan Tomov

       @generated from src/zhetrf_nopiv_gpu.cpp normal z -> d, Mon May  2 23:30:12 2016
*/
#include "magma_internal.h"
#include "trace.h"

/**
    Purpose
    =======

    DSYTRF_nopiv_gpu computes the LDLt factorization of a real symmetric
    matrix A.

    The factorization has the form
       A = U^H * D * U,   if UPLO = MagmaUpper, or
       A = L   * D * L^H, if UPLO = MagmaLower,
    where U is an upper triangular matrix, L is lower triangular, and
    D is a diagonal matrix.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U^H D U or A = L D L^H.
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -6, the GPU memory allocation failed
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.
    
    @ingroup magma_dsysv_comp
    ******************************************************************* */
extern "C" magma_int_t
magma_dsytrf_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info)
{
    #define  A(i, j)  (A)
    #define dA(i, j)  (dA +(j)*ldda + (i))
    #define dW(i, j)  (dW +(j)*ldda + (i))
    #define dWt(i, j) (dW +(j)*nb   + (i))

    /* Constants */
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    
    /* Local variables */
    bool upper = (uplo == MagmaUpper);
    magma_int_t j, k, jb, nb, ib, iinfo;

    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    /* Quick return */
    if ( n == 0 )
      return MAGMA_SUCCESS;

    nb = magma_get_dsytrf_nopiv_nb(n);
    ib = min(32, nb); // inner-block for diagonal factorization

    magma_device_t cdev;
    magma_getdevice( &cdev );

    magma_queue_t queue[2];
    magma_event_t event;
    magma_queue_create( cdev, &queue[0] );
    magma_queue_create( cdev, &queue[1] );
    magma_event_create( &event );
    trace_init( 1, 1, 2, queue );

    // CPU workspace
    double *A;
    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &A, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    // GPU workspace
    magmaDouble_ptr dW;
    if (MAGMA_SUCCESS != magma_dmalloc( &dW, (1+nb)*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /* Use hybrid blocked code. */
    if (upper) {
        //=========================================================
        // Compute the LDLt factorization A = U'*D*U without pivoting.
        // main loop
        for (j=0; j < n; j += nb) {
            jb = min(nb, (n-j));
            
            // copy A(j,j) back to CPU
            trace_gpu_start( 0, 0, "get", "get" );
            //magma_queue_wait_event( queue[1], event );
            magma_event_sync(event);
            magma_dgetmatrix_async( jb, jb, dA(j, j), ldda, A(j,j), nb, queue[1] );
            trace_gpu_end( 0, 0 );

            // factorize the diagonal block
            magma_queue_sync( queue[1] );
            trace_cpu_start( 0, "potrf", "potrf" );
            magma_dsytrf_nopiv_cpu( MagmaUpper, jb, ib, A(j, j), nb, info );
            trace_cpu_end( 0 );
            if (*info != 0) {
                *info = *info + j;
                break;
            }
            
            // copy A(j,j) back to GPU
            trace_gpu_start( 0, 0, "set", "set" );
            magma_dsetmatrix_async( jb, jb, A(j, j), nb, dA(j, j), ldda, queue[0] );
            trace_gpu_end( 0, 0 );
            
            if ( (j+jb) < n) {
                // compute the off-diagonal blocks of current block column
                trace_gpu_start( 0, 0, "trsm", "trsm" );
                magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaUnit,
                             jb, (n-j-jb),
                             c_one, dA(j, j), ldda,
                             dA(j, j+jb), ldda,
                             queue[0] );
                magma_dcopymatrix( jb, n-j-jb, dA( j, j+jb ), ldda, dWt( 0, j+jb ), nb, queue[0] );
                
                // update the trailing submatrix with D
                magmablas_dlascl_diag(MagmaUpper, jb, n-j-jb,
                                      dA(j,    j), ldda,
                                      dA(j, j+jb), ldda,
                                      queue[0], &iinfo);
                trace_gpu_end( 0, 0 );
                
                // update the trailing submatrix with U and W
                trace_gpu_start( 0, 0, "gemm", "gemm" );
                for (k=j+jb; k < n; k += nb) {
                    magma_int_t kb = min(nb,n-k);
                    magma_dgemm( MagmaConjTrans, MagmaNoTrans, kb, n-k, jb,
                                 c_neg_one, dWt(0, k), nb,
                                            dA(j, k),  ldda,
                                 c_one,     dA(k, k),  ldda,
                                 queue[0] );
                    if (k == j+jb)
                        magma_event_record( event, queue[0] );
                }
                trace_gpu_end( 0, 0 );
            }
        }
    } else {
        //=========================================================
        // Compute the LDLt factorization A = L*D*L' without pivoting.
        // main loop
        for (j=0; j < n; j += nb) {
            jb = min(nb, (n-j));
            
            // copy A(j,j) back to CPU
            trace_gpu_start( 0, 0, "get", "get" );
            //magma_queue_wait_event( queue[0], event );
            magma_event_sync(event);
            magma_dgetmatrix_async( jb, jb, dA(j, j), ldda, A(j,j), nb, queue[1] );
            trace_gpu_end( 0, 0 );
            
            // factorize the diagonal block
            magma_queue_sync( queue[1] );
            trace_cpu_start( 0, "potrf", "potrf" );
            magma_dsytrf_nopiv_cpu( MagmaLower, jb, ib, A(j, j), nb, info );
            trace_cpu_end( 0 );
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            // copy A(j,j) back to GPU
            trace_gpu_start( 0, 0, "set", "set" );
            magma_dsetmatrix_async( jb, jb, A(j, j), nb, dA(j, j), ldda, queue[0] );
            trace_gpu_end( 0, 0 );
            
            if ( (j+jb) < n) {
                // compute the off-diagonal blocks of current block column
                trace_gpu_start( 0, 0, "trsm", "trsm" );
                magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaUnit,
                             (n-j-jb), jb,
                             c_one, dA(j, j), ldda,
                             dA(j+jb, j), ldda,
                             queue[0]);
                magma_dcopymatrix( n-j-jb,jb, dA( j+jb, j ), ldda, dW( j+jb, 0 ), ldda, queue[0] );
                
                // update the trailing submatrix with D
                magmablas_dlascl_diag( MagmaLower, n-j-jb, jb,
                                       dA(j,    j), ldda,
                                       dA(j+jb, j), ldda,
                                       queue[0], &iinfo );
                trace_gpu_end( 0, 0 );
                
                // update the trailing submatrix with L and W
                trace_gpu_start( 0, 0, "gemm", "gemm" );
                for (k=j+jb; k < n; k += nb) {
                    magma_int_t kb = min(nb,n-k);
                    magma_dgemm( MagmaNoTrans, MagmaConjTrans, n-k, kb, jb,
                                 c_neg_one, dA(k, j), ldda,
                                            dW(k, 0), ldda,
                                 c_one,     dA(k, k), ldda,
                                 queue[0] );
                    if (k == j+jb)
                        magma_event_record( event, queue[0] );
                }
                trace_gpu_end( 0, 0 );
            }
        }
    }
    
    trace_finalize( "dsytrf.svg","trace.css" );
    magma_queue_destroy( queue[0] );
    magma_queue_destroy( queue[1] );
    magma_event_destroy( event );
    magma_free( dW );
    magma_free_pinned( A );
    
    return MAGMA_SUCCESS;
} /* magma_dsytrf_nopiv */
