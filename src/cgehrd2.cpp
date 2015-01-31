/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgehrd2.cpp normal z -> c, Fri Jan 30 19:00:19 2015
       
       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

/**
    Purpose
    -------
    CGEHRD2 reduces a COMPLEX general matrix A to upper Hessenberg form H by
    an orthogonal similarity transformation:  Q' * A * Q = H .

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    ilo     INTEGER
    @param[in]
    ihi     INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to CGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the N-by-N general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the orthogonal matrix Q as a product of elementary
            reflectors. See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    tau     COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
            zero.

    @param[out]
    work    (workspace) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.  LWORK >= max(1,N).
            For optimum performance LWORK >= N*NB, where NB is the
            optimal blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ---------------
    The matrix Q is represented as a product of (ihi-ilo) elementary
    reflectors

       Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
    exit in A(i+2:ihi,i), and tau in TAU(i).

    The contents of A are illustrated by the following example, with
    n = 7, ilo = 2 and ihi = 6:

    @verbatim
    on entry,                        on exit,

    ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
    (                         a )    (                          a )
    @endverbatim

    where a denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an
    element of the vector defining H(i).

    This implementation follows the hybrid algorithm and notations described in

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.

    @ingroup magma_cgeev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgehrd2(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info)
{
    #define A(i_,j_)  (A  + (i_) + (j_)*lda)
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    
    magmaFloatComplex c_one = MAGMA_C_ONE;
    magmaFloatComplex c_zero = MAGMA_C_ZERO;

    magma_int_t nb = magma_get_cgehrd_nb(n);
    magma_int_t ldda = ((n+31)/32)*32;

    magma_int_t i, nh, iws;
    magma_int_t iinfo;
    magma_int_t ldwork;
    magma_int_t lquery;

    *info = 0;
    iws = n*nb;
    work[0] = MAGMA_C_MAKE( iws, 0 );

    lquery = (lwork == -1);
    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > max(1,n)) {
        *info = -2;
    } else if (ihi < min(ilo,n) || ihi > n) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if (lwork < max(1,n) && ! lquery) {
        *info = -8;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    // Adjust from 1-based indexing
    ilo -= 1;
    
    // Quick return if possible
    nh = ihi - ilo;
    if (nh <= 1) {
        work[0] = c_one;
        return *info;
    }

    // If not enough workspace, use unblocked code
    if ( lwork < iws ) {
        nb = 1;
    }

    if (nb == 1 || nb > nh) {
        // Use unblocked code below
        i = ilo;
    }
    else {
        // Use blocked code
        
        // GPU workspace is:
        //   nb*ldda for dwork for clahru
        //   nb*ldda for dV
        //   n*ldda  for dA
        //   nb*nb   for dT
        magmaFloatComplex *dwork;
        if (MAGMA_SUCCESS != magma_cmalloc( &dwork, 2*nb*ldda + n*ldda + nb*nb )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        magmaFloatComplex *dV = dwork + nb*ldda;
        magmaFloatComplex *dA = dwork + nb*ldda*2;
        magmaFloatComplex *dT = dwork + nb*ldda*2 + n*ldda;
        ldwork = ldda;
            
        magmaFloatComplex *T;
        magma_cmalloc_cpu( &T, nb*nb );
        if ( T == NULL ) {
            magma_free( dwork );
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
    
        // zero first block of V, which is lower triangular
        magmablas_claset( MagmaFull, nb, nb, c_zero, c_zero, dV, ldda );
    
        // Set elements 0:ILO-1 and IHI-1:N-2 of TAU to zero
        for (i = 0; i < ilo; ++i)
            tau[i] = c_zero;
    
        for (i = max(0,ihi-1); i < n-1; ++i)
            tau[i] = c_zero;
    
        assert( nb % 4 == 0 );
        for (i=0; i < nb*nb; i += 4)
            T[i] = T[i+1] = T[i+2] = T[i+3] = c_zero;
    
        // Copy the matrix to the GPU
        magma_csetmatrix( n, n-ilo, A(0,ilo), lda, dA, ldda );
        
        for (i = ilo; i < ihi-1 - nb; i += nb) {
            // Reduce columns i:i+nb-1 to Hessenberg form, returning the
            // matrices V and T of the block reflector H = I - V*T*V'
            // which performs the reduction, and also the matrix Y = A*V*T
            
            // Get the current panel (no need for the 1st iteration)
            magma_cgetmatrix( ihi-i, nb,
                              dA(i,i-ilo), ldda,
                              A(i,i), lda );
            
            // add 1 to i for 1-based index
            magma_clahr2( ihi, i+1, nb,
                          dA(0,i-ilo), ldda,
                          dV,          ldda,
                          A(0,i),      lda,
                          &tau[i], T, nb, work, ldwork );
            
            // Copy T from the CPU to dT on the GPU
            magma_csetmatrix( nb, nb, T, nb, dT, nb );
            
            magma_clahru( n, ihi, i, nb,
                          A(0,i),      lda,
                          dA(0,i-ilo), ldda, // dA
                          dA(i,i-ilo), ldda, // dY, stored over current panel
                          dV,          ldda,
                          dT, dwork );
        }
        
        // Copy remainder to host
        magma_cgetmatrix( n, n-i,
                          dA(0,i-ilo), ldda,
                          A(0,i),  lda );
        
        magma_free( dwork );
        magma_free_cpu( T );
    }

    // Use unblocked code to reduce the rest of the matrix
    // add 1 to i for 1-based index
    i += 1;
    lapackf77_cgehd2(&n, &i, &ihi, A, &lda, tau, work, &iinfo);
    work[0] = MAGMA_C_MAKE( iws, 0 );

    return *info;
} /* magma_cgehrd2 */
