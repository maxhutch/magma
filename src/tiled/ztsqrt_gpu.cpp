/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ztsqrt computes a QR factorization of a rectangular matrix
    formed by coupling a complex N-by-N upper triangular tile A1
    on top of a complex M-by-N tile A2:
    
       | A1 | = Q * R
       | A2 |
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of columns of the tile A2. M >= 0.
    
    @param[in]
    n       INTEGER
            The number of rows of the tile A1.
            The number of columns of the tiles A1 and A2. N >= 0.
    
    @param[in,out]
    A1      COMPLEX_16 array on the GPU, dimension (LDA,N)
            On entry, the N-by-N tile A1.
            On exit, the elements on and above the diagonal of the array
            contain the N-by-N upper trapezoidal tile R;
            the elements below the diagonal are not referenced.
    
    @param[in,out]
    A2      COMPLEX_16 array on the GPU, dimension (LDA,N)
            On entry, the M-by-N tile A2.
            On exit, all the elements, with the array TAU, represent
            the unitary tile Q as a product of elementary reflectors
            (see Further Details).
    
    @param[in]
    lda     INTEGER
            The leading dimension of the tile A1 and A2. LDA >= max(1,M).
    
    @param[out]
    tau     INTEGER
            The scalar factors of the elementary reflectors (see Further
            Details).
    
    @param[out]
    work    COMPLEX_16 array on the CPU host, dimension (LWORK).
        
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK. TODO: LWORK >= ???.
    
    @param[out]
    dwork   COMPLEX_16 array on the GPU, dimension TODO.
        
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_zgeqrf_tile
    ********************************************************************/
extern "C" magma_int_t
magma_ztsqrt_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A1, magmaDoubleComplex *A2, magma_int_t  lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dwork,
    magma_int_t *info )
{
    #define A1(a_1,a_2) (A1 + (a_2)*lda + (a_1))
    #define A2(a_1,a_2) (A2 + (a_2)*lda + (a_1))
    #define t_ref(a_1)  (dwork + (a_1))
    #define d_ref(a_1)  (dwork + (lddwork+(a_1))*nb)
    #define dd_ref(a_1) (dwork + (2*lddwork+(a_1))*nb)
    #define work_A1     (work)
    #define work_A2     (work + nb)
    #define hwork       (work + (nb)*m)
    
    magma_int_t i, k, ldwork, lddwork, old_i, old_ib, rows, cols;
    magma_int_t nbmin, ib, ldda;
    
    /* Function Body */
    *info = 0;
    magma_int_t nb = magma_get_zgeqrf_nb( m );
    
    magma_int_t lwkopt = (n + m) * nb;
    work[0] = MAGMA_Z_MAKE( lwkopt, 0 );
    magma_int_t lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    } else if (lwork < max(1,n) && ! lquery) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;
    
    k = min(m,n);
    if (k == 0) {
        work[0] = MAGMA_Z_MAKE( 1, 0 );
        return *info;
    }
    
    //magma_int_t lhwork = lwork - m*nb;
    
    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    
    ldda = m;
    nbmin = 2;
    ldwork = m;
    lddwork= k;
    
    // This is only blocked code for now
    for (i = 0; i < k; i += nb) {
        ib = min(k-i, nb);
        rows = m -i;
        rows = m;
        // Send the next panel (diagonal block of A1 & block column of A2)
        // to the CPU (in work_A1 and work_A2)
        magma_zgetmatrix_async( rows, ib,
                                A2(0,i), lda,
                                work_A2,     ldwork, stream[1] );
        
                            // A1(i,i), lda*sizeof(magmaDoubleComplex),
                            // the diagonal of A1 is in d_ref generated and
                            // passed from magma_zgeqrf_gpu
        magma_zgetmatrix_async( ib, ib,
                                d_ref(i), ib,
                                work_A1,  ldwork, stream[1] );
        
        if (i > 0) {
            /* Apply H' to A(i:m,i+2*ib:n) from the left */
            // update T2
            cols = n - old_i - 2*old_ib;
            magma_zssrfb(m, cols, &old_ib,
                         A2(    0, old_i), lda, t_ref(old_i), &lddwork,
                         A1(old_i, old_i+2*old_ib), lda,
                         A2(    0, old_i+2*old_ib), lda,
                         dd_ref(0), &lddwork);
        }
        
        magma_queue_sync( stream[1] );
        
        // TTT - here goes the CPU PLASMA code
        //       Matrix T has to be put in hwork with lda = ib and 0s
        //       in the parts that are not used - copied on GPU in t_ref(i)
        
        // Now diag of A1 is updated, send it back asynchronously to the GPU.
        // We have to play interchaning these copies to see which is faster
        magma_zsetmatrix_async( ib, ib,
                                work_A1,  ib,
                                d_ref(i), ib, stream[0] );
        // Send the panel from A2 back to the GPU
        magma_zsetmatrix( m, ib, work_A2, ldwork, A2(0,i), lda );
        
        if (i + ib < n) {
            // Send the triangular factor T from hwork to the GPU in t_ref(i)
            magma_zsetmatrix( ib, ib, hwork, ib, t_ref(i), lddwork );
            
            if (i+nb < k) {
                /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                // if we can do one more step, first update T1
                magma_zssrfb(m, ib, &ib,
                             A2(0, i),    lda, t_ref(i), &lddwork,
                             A1(i, i+ib), lda,
                             A2(0, i+ib), lda,
                             dd_ref(0), &lddwork);
            }
            else {
                cols = n - i - ib;
                // otherwise, update until the end and fix the panel
                magma_zssrfb(m, cols, &ib,
                             A2(0, i),    lda, t_ref(i), &lddwork,
                             A1(i, i+ib), lda,
                             A2(0, i+ib), lda,
                             dd_ref(0), &lddwork);
            }
            old_i = i;
            old_ib = ib;
        }
    }
    
    return *info;
} /* magma_ztsqrt_gpu */

#undef A1
#undef A2
#undef t_ref
#undef d_ref
#undef dd_ref
#undef hwork
#undef work_A1
#undef work_A2
