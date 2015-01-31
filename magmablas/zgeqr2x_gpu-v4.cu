/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "commonblas_z.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/**
    Purpose
    -------
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    This expert routine requires two more arguments than the standard
    zgeqr2, namely, dT and ddA, explained below. The storage for A is
    also not as in the LAPACK's zgeqr2 routine (see below).

    The first is used to output the triangular
    n x n factor T of the block reflector used in the factorization.
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R. This routine implements the left looking QR.

    This version adds internal blocking.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).
    \n
            the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    dtau    COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      COMPLEX_16 array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector
            used in the factorization. The lower triangular part is 0.

    @param[out]
    ddA     COMPLEX_16 array, dimension N x N.
            Stores the elements of the upper N x N diagonal block of A.
            LAPACK stores this array in A. There are 0s below the diagonal.

    @param
    dwork   (workspace) DOUBLE_PRECISION array, dimension (3 N)

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v**H

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_zgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr dwork,
    magma_queue_t queue,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (j_)*(ldda) + (i_))
    #define dT(i_,j_) (dT + (j_)*(k)    + (i_))
    #define BS 32

    magma_int_t i, k;

    double *dnorm = (double *)dwork;
    magmaDoubleComplex *work = (magmaDoubleComplex *)(dwork+2*n);

    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);
    magmablasSetKernelStream(queue);

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(m,n);
    magmablas_dznrm2_cols(m, k, dA(0,0), ldda, dnorm);

    for (magma_int_t b=0; b < k; b += BS) {
        for (i = b; i < min(k, b+BS); ++i) {

            /*   Apply H**H to A(:,i) from the left */
            if (i-b > 0) {
                /* Compute the (i-1)th column of T */
                if ( i-1 > 0 ) {
                    magma_zgemv_kernel3<<< i-1, BLOCK_SIZE, 0, magma_stream >>>
                        ( m-i+1, dA(i-1,0), ldda, dA(i-1, i-1), work, dtau+i-1);
                    magma_ztrmv_kernel2<<< i-1, i-1, 0, magma_stream >>>
                        ( dT(0,0), k, work, dT(0,i-1), dtau+i-1);
                }

                /* dwork = V**H c */
                magma_zgemv_kernel1<<< i-b, BLOCK_SIZE, 0, magma_stream >>>
                    (m-b, dA(b, b),  ldda, dA(b,i), work);

                /* dwork = T**H work */
                magma_ztrmv_tkernel<<< i-b, i-b, 0, magma_stream >>>
                    (dT(b,b), k, work, work+i-b);

                /* c = c - V work */
                if ( m-b > 0 ) {
                    dim3  blocks3( (m-b + BLOCK_SIZE-1) / BLOCK_SIZE );
                    dim3 threads3( BLOCK_SIZE );
                    magma_zgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>
                        (m-b, i-b, dA(b,b), ldda,  work+i-b, dA(b, i));
                }
            }

            /*   Adjust the dnorm[i] to hold the norm of A(i:m,i) */
            if ( i > 0 ) {
                magma_dznrm2_adjust_kernel<<< 1, i, 0, magma_stream >>>(dnorm+i, dA(0, i));
            }

            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i)
                1. 1 is not yet put on the diagonal of A
                2. Elements above the diagonal are copied in ddA and
                   the ones in A are set to zero
                3. update T */
            magma_zlarfgx_gpu(m-i, dA(i, i), dA(min(i+1,m),i), dtau+i,
                              dnorm+i, ddA + i + i*n, i);

            if (i == 0) {
                magmaDoubleComplex tt = MAGMA_Z_ONE;
                magmablas_zlacpy(MagmaUpperLower, 1, 1, dtau, 1, dT(0,0), 1);
                magma_zsetmatrix_async(1, 1, &tt, 1, dA(i, i), 1, magma_stream);
            }
        }
        if ( i-1 > 0 ) {
            magma_zgemv_kernel3<<< i-1, BLOCK_SIZE, 0, magma_stream >>>
                ( m-i+1, dA(i-1,0), ldda, dA(i-1, i-1), work, dtau+i-1);
            magma_ztrmv_kernel2<<< i-1, i-1, 0, magma_stream >>>
                ( dT(0,0), k, work, dT(0,i-1), dtau+i-1);
        }

        /* Apply the transformations to the trailing matrix. */
        //magma_zlarfb2_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
        magma_zlarfb2_gpu(
                           m-b, k-i, BS,
                           dA(b, b), ldda, dT+b+b*k, k,
                           dA(b, i), ldda, work, k-i);
    }

    magmablasSetKernelStream(cstream);

    return *info;
} /* magma_zgeqr2 */
