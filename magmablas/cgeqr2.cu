/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zgeqr2.cu normal z -> c, Fri Jan 30 19:00:10 2015

*/
#include "common_magma.h"


/**
    Purpose
    -------
    CGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R
    using the non-blocking Householder QR.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    dtau    COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param
    dwork   (workspace) DOUBLE_PRECISION array, dimension (N)

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

    @ingroup magma_cgeqrf_aux
    ********************************************************************/
extern "C" magma_int_t
magma_cgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dwork,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*(ldda))
    
    magma_int_t i, k;

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

    /* Workspace for diagonal entries - restored at the end */
    magmaFloatComplex *Aks;
    magma_cmalloc( &Aks, k );
    if ( Aks == NULL ) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(*info) );
    }
    else {
        for (i = 0; i < k; ++i) {
            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
            magma_clarfg_gpu(m-i, dA(i, i), dA(min(i+1,m), i), dtau+i, dwork, &Aks[i]);

            if (n-i-1 > 0) {
               /* Apply H(i)' to A(i:m,i+1:n) from the left */
               magma_clarf_gpu(m-i, n-i-1, dA(i, i), dtau+i, dA(i, i+1), ldda);
            }
        }

        magma_ccopymatrix( 1, k, Aks, 1, dA(0, 0), ldda+1 );
    }
    
    magma_free(Aks);
    return *info;
} /* magma_cgeqr2 */
