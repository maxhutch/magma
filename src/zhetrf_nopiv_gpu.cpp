/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "trace.h"

// === Define what BLAS to use ============================================
#define PRECISION_z
#if (defined(PRECISION_s) || defined(PRECISION_d))
  #define cublasZgemm magmablas_zgemm
  #define cublasZtrsm magmablas_ztrsm
#endif

#if (GPUSHMEM >= 200)
#if (defined(PRECISION_s))
     #undef  cublasSgemm
     #define cublasSgemm magmablas_sgemm_fermi80
  #endif
#endif
// === End defining what BLAS to use ======================================

#define  A(i, j)  (A)
#define dA(i, j)  (dA +(j)*ldda + (i))
#define dW(i, j)  (dW +(j)*ldda + (i))
#define dWt(i, j) (dW +(j)*nb   + (i))

extern "C" magma_int_t 
magma_zhetrf_nopiv_gpu(magma_uplo_t uplo, magma_int_t n, 
                       magmaDoubleComplex *dA, magma_int_t ldda,
                       magma_int_t *info)
{
/*  -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   

    ZHETRF_nopiv computes the LDLt factorization of a complex Hermitian   
    matrix A. This version does not require work space on the GPU passed 
    as input. GPU memory is allocated in the routine.

    The factorization has the form   
       A = U\*\*H * D * U,  if UPLO = 'U', or   
       A = L  * D * L\*\*H, if UPLO = 'L',   
    where U is an upper triangular matrix, L is lower triangular, and
    D is a diagonal matrix.

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U\*\*H*U or A = L*L\*\*H.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  if INFO = -6, the GPU memory allocation failed 
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================    */


    /* Local variables */
    magmaDoubleComplex zone  = MAGMA_Z_ONE;
    magmaDoubleComplex mzone = MAGMA_Z_NEG_ONE;
    int                upper = (uplo == MagmaUpper);
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

    nb = magma_get_zhetrf_nopiv_nb(n);
    ib = min(32, nb); // inner-block for diagonal factorization

    magma_queue_t stream[2];
    magma_event_t event;
    magma_queue_create(&stream[0]);
    magma_queue_create(&stream[1]);
    magma_event_create( &event );
    trace_init( 1, 1, 2, (CUstream_st**)stream );

    // CPU workspace
    magmaDoubleComplex *A;
    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &A, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    // GPU workspace
    magmaDoubleComplex *dW;
    if (MAGMA_SUCCESS != magma_zmalloc( &dW, (1+nb)*ldda )) {
          *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
    }
    //if (nb <= 1 || nb >= n) 
    //{
    //    lapackf77_zpotrf(uplo_, &n, a, &lda, info);
    //} else 
    {
        /* Use hybrid blocked code. */
        if (upper) {
            //=========================================================
            // Compute the LDLt factorization A = U'*D*U without pivoting.
            // main loop
            for (j=0; j<n; j += nb) {
                jb = min(nb, (n-j));

                // copy A(j,j) back to CPU
                trace_gpu_start( 0, 0, "get", "get" );
                magma_zgetmatrix_async(jb, jb, dA(j, j), ldda, A(j,j), nb, stream[0]);
                trace_gpu_end( 0, 0 );

                // factorize the diagonal block
                magma_queue_sync(stream[0]);
                trace_cpu_start( 0, "potrf", "potrf" );
                zhetrf_nopiv_cpu(MagmaUpper, jb, ib, A(j, j), nb, info);
                trace_cpu_end( 0 );
                if (*info != 0){
                    *info = *info + j;
                    break;
                }

                // copy A(j,j) back to GPU
                trace_gpu_start( 0, 0, "set", "set" );
                magma_zsetmatrix_async(jb, jb, A(j, j), nb, dA(j, j), ldda, stream[0]);
                trace_gpu_end( 0, 0 );
                
                if ( (j+jb) < n) {
                    // compute the off-diagonal blocks of current block column
                    magmablasSetKernelStream( stream[0] );
                    trace_gpu_start( 0, 0, "trsm", "trsm" );
                    magma_ztrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaUnit, 
                                jb, (n-j-jb), 
                                zone, dA(j, j),    ldda, 
                                      dA(j, j+jb), ldda);
                    magma_zcopymatrix( jb, n-j-jb, dA( j, j+jb ), ldda, dWt( 0, j+jb ), nb );

                    // update the trailing submatrix with D
                    magmablas_zlascl_diag(MagmaUpper, jb, n-j-jb,
                                          dA(j,    j), ldda,
                                          dA(j, j+jb), ldda,
                                          &iinfo);
                    magma_event_record( event, stream[0] );
                    trace_gpu_end( 0, 0 );

                    // update the trailing submatrix with U and W
                    trace_gpu_start( 0, 0, "gemm", "gemm" );
                    for (k=j+jb; k<n; k+=nb)
                    {
                        magma_int_t kb = min(nb,n-k);
                        magma_zgemm(MagmaConjTrans, MagmaNoTrans, kb, n-k, jb,
                                    mzone, dWt(0, k), nb, 
                                           dA(j, k), ldda,
                                    zone,  dA(k, k), ldda);
                    }
                    trace_gpu_end( 0, 0 );
                }
            }
        } else {
            //=========================================================
            // Compute the LDLt factorization A = L*D*L' without pivoting.
            // main loop
            for (j=0; j<n; j+=nb) {
                jb = min(nb, (n-j));

                // copy A(j,j) back to CPU
                trace_gpu_start( 0, 0, "get", "get" );
                magma_zgetmatrix_async(jb, jb, dA(j, j), ldda, A(j,j), nb, stream[0]);
                trace_gpu_end( 0, 0 );

                // factorize the diagonal block
                magma_queue_sync(stream[0]);
                trace_cpu_start( 0, "potrf", "potrf" );
                zhetrf_nopiv_cpu(MagmaLower, jb, ib, A(j, j), nb, info);
                trace_cpu_end( 0 );
                if (*info != 0){
                    *info = *info + j;
                    break;
                }
                // copy A(j,j) back to GPU
                trace_gpu_start( 0, 0, "set", "set" );
                magma_zsetmatrix_async(jb, jb, A(j, j), nb, dA(j, j), ldda, stream[0]);
                trace_gpu_end( 0, 0 );
                
                if ( (j+jb) < n) {
                    // compute the off-diagonal blocks of current block column
                    magmablasSetKernelStream( stream[0] );
                    trace_gpu_start( 0, 0, "trsm", "trsm" );
                    magma_ztrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaUnit, 
                                (n-j-jb), jb, 
                                zone, dA(j,    j), ldda, 
                                      dA(j+jb, j), ldda);
                    magma_zcopymatrix( n-j-jb,jb, dA( j+jb, j ), ldda, dW( j+jb, 0 ), ldda );

                    // update the trailing submatrix with D
                    magmablas_zlascl_diag(MagmaLower, n-j-jb, jb,
                                          dA(j,    j), ldda,
                                          dA(j+jb, j), ldda,
                                          &iinfo);
                    magma_event_record( event, stream[0] );
                    trace_gpu_end( 0, 0 );

                    // update the trailing submatrix with L and W
                    trace_gpu_start( 0, 0, "gemm", "gemm" );
                    for (k=j+jb; k<n; k+=nb)
                    {
                        magma_int_t kb = min(nb,n-k);
                        magma_zgemm(MagmaNoTrans, MagmaConjTrans, n-k, kb, jb,
                                    mzone, dA(k, j), ldda, 
                                           dW(k, 0), ldda,
                                    zone,  dA(k, k), ldda);
                    }
                    trace_gpu_end( 0, 0 );
                }
            }
        }
    }
    
    trace_finalize( "zhetrf.svg","trace.css" );
    magma_queue_destroy(stream[0]);
    magma_queue_destroy(stream[1]);
    magma_event_destroy( event );
    magma_free( dW );
    magma_free_pinned( A );
    
    return MAGMA_SUCCESS;
} /* magma_zhetrf_nopiv */

