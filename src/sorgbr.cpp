/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates

       @generated from src/zungbr.cpp normal z -> s, Mon May  2 23:30:24 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    SORGBR generates one of the real orthogonal matrices Q or P**H
    determined by SGEBRD when reducing a real matrix A to bidiagonal
    form: A = Q * B * P**H.  Q and P**H are defined as products of
    elementary reflectors H(i) or G(i) respectively.
   
    If VECT = MagmaQ, A is assumed to have been an M-by-K matrix, and Q
    is of order M:
    if m >= k, Q = H(1) H(2) . . . H(k) and SORGBR returns the first n
    columns of Q, where m >= n >= k;
    if m < k, Q = H(1) H(2) . . . H(m-1) and SORGBR returns Q as an
    M-by-M matrix.
   
    If VECT = MagmaP, A is assumed to have been a K-by-N matrix, and P**H
    is of order N:
    if k < n, P**H = G(k) . . . G(2) G(1) and SORGBR returns the first m
    rows of P**H, where n >= m >= k;
    if k >= n, P**H = G(n-1) . . . G(2) G(1) and SORGBR returns P**H as
    an N-by-N matrix.

    Arguments
    ---------
    @param[in]
    vect    magma_vect_t
            Specifies whether the matrix Q or the matrix P**H is
            required, as defined in the transformation applied by SGEBRD:
            = MagmaQ:  generate Q;
            = MagmaP:  generate P**H.
   
    @param[in]
    m       magma_int_t
            The number of rows of the matrix Q or P**H to be returned.
            M >= 0.
   
    @param[in]
    n       magma_int_t
            The number of columns of the matrix Q or P**H to be returned.
            N >= 0.
            If VECT = MagmaQ, M >= N >= min(M,K);
            if VECT = MagmaP, N >= M >= min(N,K).
   
    @param[in]
    k       magma_int_t
            If VECT = MagmaQ, the number of columns in the original M-by-K
            matrix reduced by SGEBRD.
            If VECT = MagmaP, the number of rows in the original K-by-N
            matrix reduced by SGEBRD.
            K >= 0.
   
    @param[in,out]
    A       float array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by SGEBRD.
            On exit, the M-by-N matrix Q or P**H.
   
    @param[in]
    lda     magma_int_t
            The leading dimension of the array A. LDA >= M.
   
    @param[in]
    tau     float array, dimension
                                  (min(M,K)) if VECT = MagmaQ
                                  (min(N,K)) if VECT = MagmaP
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i), which determines Q or P**H, as
            returned by SGEBRD in its array argument TAUQ or TAUP.
   
    @param[out]
    work    float array, dimension (MAX(1,LWORK))
            On exit, if *info = 0, WORK(1) returns the optimal LWORK.
   
    @param[in]
    lwork   magma_int_t
            The dimension of the array WORK. LWORK >= max(1,min(M,N)).
            For optimum performance LWORK >= min(M,N)*NB, where NB
            is the optimal blocksize.
   
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.
   
    @param[out]
    info    magma_int_t
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_sgesvd_comp
    ********************************************************************/
extern "C" magma_int_t
magma_sorgbr(
    magma_vect_t vect, magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info)
{
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    
    // Constants
    const float c_zero = MAGMA_S_ZERO;
    const float c_one  = MAGMA_S_ONE;
    
    // Local variables
    bool lquery, wantq;
    magma_int_t i, iinfo, j, lwkopt, mn;
    
    // Test the input arguments
    *info = 0;
    wantq = (vect == MagmaQ);
    mn = min( m, n );
    lquery = (lwork == -1);
    if ( ! wantq && vect != MagmaP ) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0 || (wantq && (n > m || n < min(m,k))) || ( ! wantq && (m > n || m < min(n,k)))) {
        *info = -3;
    } else if (k < 0) {
        *info = -4;
    } else if (lda < max( 1, m )) {
        *info = -6;
    }

    // Check workspace size
    if (*info == 0) {
        work[0] = c_one;
        if (wantq) {
            if (m >= k) {
                // magma_sorgqr takes dT instead of work
                // magma_sorgqr2 doesn't take work
                //magma_sorgqr2( m, n, k, A, lda, tau, work, -1, &iinfo );
                work[0] = c_one;
            }
            else if (m > 1) {
                //magma_sorgqr2( m-1, m-1, m-1, A(1,1), lda, tau, work, -1, &iinfo );
                work[0] = c_one;
            }
        }
        else {
            if (k < n) {
                magma_sorglq( m, n, k, A, lda, tau, work, -1, &iinfo );
            }
            else if (n > 1) {
                magma_sorglq( n-1, n-1, n-1, A(1,1), lda, tau, work, -1, &iinfo );
            }
        }
        lwkopt = MAGMA_S_REAL( work[0] );
        lwkopt = max( lwkopt, mn );
        if (lwork < lwkopt && ! lquery) {
            *info = -9;
        }
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        work[0] = magma_smake_lwork( lwkopt );
        return *info;
    }

    // Quick return if possible
    if (m == 0  ||  n == 0) {
        work[0] = c_one;
        return *info;
    }

    if (wantq) {
        // Form Q, determined by a call to SGEBRD to reduce an m-by-k
        // matrix
        if (m >= k) {
            // If m >= k, assume m >= n >= k
            magma_sorgqr2( m, n, k, A, lda, tau, /*work, lwork,*/ &iinfo );
        }
        else {
            // If m < k, assume m = n
    
            // Shift the vectors which define the elementary reflectors one
            // column to the right, and set the first row and column of Q
            // to those of the unit matrix
            for (j=m-1; j >= 1; --j) {
                *A(0,j) = c_zero;
                for (i=j + 1; i < m; ++i) {
                    *A(i,j) = *A(i,j-1);
                }
            }
            *A(0,0) = c_one;
            for (i=1; i < m; ++i) {
                *A(i,0) = c_zero;
            }
            if (m > 1) {
                // Form Q(2:m,2:m)
                magma_sorgqr2( m-1, m-1, m-1, A(1,1), lda, tau, /*work, lwork,*/ &iinfo );
            }
        }
    }
    else {
        // Form P**H, determined by a call to SGEBRD to reduce a k-by-n
        // matrix
        if (k < n) {
            // If k < n, assume k <= m <= n
            magma_sorglq( m, n, k, A, lda, tau, work, lwork, &iinfo );
        }
        else {
            // If k >= n, assume m = n
            
            // Shift the vectors which define the elementary reflectors one
            // row downward, and set the first row and column of P**H to
            // those of the unit matrix
            *A(0,0) = c_one;
            for (i=1; i < n; ++i) {
                *A(i,0) = c_zero;
            }
            for (j=1; j < n; ++j) {
                for (i=j-1; i >= 1; --i) {
                    *A(i,j) = *A(i-1,j);
                }
                *A(0,j) = c_zero;
            }
            if (n > 1) {
                // Form P**H(2:n,2:n)
                magma_sorglq( n-1, n-1, n-1, A(1,1), lda, tau, work, lwork, &iinfo );
            }
        }
    }
    
    work[0] = magma_smake_lwork( lwkopt );
    return *info;
}
