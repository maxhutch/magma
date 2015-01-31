/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlahr2_m.cpp normal z -> d, Fri Jan 30 19:00:19 2015
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_d

/**
    Purpose
    -------
    DLAHR2 reduces the first NB columns of a real general n-BY-(n-k+1)
    matrix A so that elements below the k-th subdiagonal are zero. The
    reduction is performed by an orthogonal similarity transformation
    Q' * A * Q. The routine returns the matrices V and T which determine
    Q as a block reflector I - V*T*V', and also the matrix Y = A * V.
    (Note this is different than LAPACK, which computes Y = A * V * T.)

    This is an auxiliary routine called by DGEHRD.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.

    @param[in]
    k       INTEGER
            The offset for the reduction. Elements below the k-th
            subdiagonal in the first NB columns are reduced to zero.
            K < N.

    @param[in]
    nb      INTEGER
            The number of columns to be reduced.

    @param[in,out]
    A       DOUBLE_PRECISION array, dimension (LDA,N-K+1)
            On entry, the n-by-(n-k+1) general matrix A.
            On exit, the elements on and above the k-th subdiagonal in
            the first NB columns are overwritten with the corresponding
            elements of the reduced matrix; the elements below the k-th
            subdiagonal, with the array TAU, represent the matrix Q as a
            product of elementary reflectors. The other columns of A are
            unchanged. See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    tau     DOUBLE_PRECISION array, dimension (NB)
            The scalar factors of the elementary reflectors. See Further
            Details.

    @param[out]
    T       DOUBLE_PRECISION array, dimension (LDT,NB)
            The upper triangular matrix T.

    @param[in]
    ldt     INTEGER
            The leading dimension of the array T.  LDT >= NB.

    @param[out]
    Y       DOUBLE_PRECISION array, dimension (LDY,NB)
            The n-by-nb matrix Y.

    @param[in]
    ldy     INTEGER
            The leading dimension of the array Y. LDY >= N.

    @param[in,out]
    data    Structure with pointers to dA, dT, dV, dW, dY
            which are distributed across multiple GPUs.

    Further Details
    ---------------
    The matrix Q is represented as a product of nb elementary reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i+k-1) = 0, v(i+k) = 1; v(i+k+1:n) is stored on exit in
    A(i+k+1:n,i), and tau in TAU(i).

    The elements of the vectors v together form the (n-k+1)-by-nb matrix
    V which is needed, with T and Y, to apply the transformation to the
    unreduced part of the matrix, using an update of the form:
    A := (I - V*T*V') * (A - Y*T*V').

    The contents of A on exit are illustrated by the following example
    with n = 7, k = 3 and nb = 2:

    @verbatim
       ( a   a   a   a   a )
       ( a   a   a   a   a )
       ( a   a   a   a   a )
       ( h   h   a   a   a )
       ( v1  h   a   a   a )
       ( v1  v2  a   a   a )
       ( v1  v2  a   a   a )
    @endverbatim

    where "a" denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an
    element of the vector defining H(i).

    This implementation follows the hybrid algorithm and notations described in

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.

    @ingroup magma_dgeev_aux
    ********************************************************************/
extern "C" magma_int_t
magma_dlahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t ldt,
    double *Y, magma_int_t ldy,
    struct dgehrd_data *data )
{
    #define  A(  i, j ) ( A + (i) + (j)*lda)
    #define  Y(  i, j ) ( Y + (i) + (j)*ldy)
    #define  T(  i, j ) ( T + (i) + (j)*ldt)
    #define dA(  d, i, j ) (data->A [d] + (i) + (j)*ldda)
    #define dTi( d       ) (data->Ti[d])
    #define dV(  d, i, j ) (data->V [d] + (i) + (j)*ldv )
    #define dVd( d, i, j ) (data->Vd[d] + (i) + (j)*ldvd)
    #define dY(  d, i, j ) (data->Y [d] + (i) + (j)*ldda)

    double c_zero    = MAGMA_D_ZERO;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double tmp;

    magma_int_t ngpu = data->ngpu;
    magma_int_t ldda = data->ldda;
    magma_int_t ldv  = data->ldv;
    magma_int_t ldvd = data->ldvd;
    
    magma_int_t ione = 1;
    
    magma_int_t d, dki1, dn, nblocks, gblock, lblock, lgid;
    magma_int_t n_k_i_1, n_k;
    double scale;

    magma_int_t i;
    double ei = MAGMA_D_ZERO;

    magma_int_t info_data = 0;
    magma_int_t *info = &info_data;
    if (n < 0) {
        *info = -1;
    } else if (k < 0 || k >= n) {
        *info = -2;
    } else if (nb < 1 || nb > n) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if (ldt < nb) {
        *info = -8;
    } else if (ldy < max(1,n)) {
        *info = -10;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    // adjust from 1-based indexing
    k -= 1;

    // Function Body
    if (n <= 1)
        return *info;
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    // zero out current top block of V on all GPUs
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magmablasSetKernelStream( data->streams[d] );
        magmablas_dlaset( MagmaFull, nb, nb, c_zero, c_zero, dV(d,k,0), ldv );
    }
    
    // set all Y=0
    lapackf77_dlaset( "Full", &n, &nb, &c_zero, &c_zero, Y, &ldy );
    
    for (i = 0; i < nb; ++i) {
        n_k_i_1 = n - k - i - 1;
        n_k     = n - k;
        
        if (i > 0) {
            // Finish applying I - V * T * V' on right
            tmp = MAGMA_D_NEGATE( tau[i-1] );
            blasf77_daxpy( &n_k, &tmp, Y(k,i-1), &ione, A(k,i), &ione );
            
            // Apply I - V * T' * V' to this column (call it b) from the
            // left, using the last column of T as workspace, w.
            //
            // Let  V = ( V1 )   and   b = ( b1 )   (first i-1 rows)
            //          ( V2 )             ( b2 )
            // where V1 is unit lower triangular
            
            // w := b1 = A(k+1:k+i, i)
            blasf77_dcopy( &i,
                           A(k+1,i), &ione,
                           T(0,nb-1), &ione );
            
            // w := V1' * b1 = VA(k+1:k+i, 0:i-1)' * w
            blasf77_dtrmv( "Lower", "Conj", "Unit", &i,
                           A(k+1,0), &lda,
                           T(0,nb-1), &ione );
            
            // w := w + V2'*b2 = w + VA(k+i+1:n-1, 0:i-1)' * A(k+i+1:n-1, i)
            blasf77_dgemv( "Conj", &n_k_i_1, &i,
                           &c_one, A(k+i+1,0), &lda,
                                   A(k+i+1,i), &ione,
                           &c_one, T(0,nb-1), &ione );
            
            // w := T'*w = T(0:i-1, 0:i-1)' * w
            blasf77_dtrmv( "Upper", "Conj", "Non-unit", &i,
                           T(0,0), &ldt,
                           T(0,nb-1), &ione );
            
            // b2 := b2 - V2*w = A(k+i+1:n-1, i) - VA(k+i+1:n-1, 0:i-1) * w
            blasf77_dgemv( "No trans", &n_k_i_1, &i,
                           &c_neg_one, A(k+i+1,0), &lda,
                                       T(0,nb-1), &ione,
                           &c_one,     A(k+i+1,i), &ione );
            
            // w := V1*w = VA(k+1:k+i, 0:i-1) * w
            blasf77_dtrmv( "Lower", "No trans", "Unit", &i,
                           A(k+1,0), &lda,
                           T(0,nb-1), &ione );
            
            // b1 := b1 - w = A(k+1:k+i-1, i) - w
            blasf77_daxpy( &i,
                           &c_neg_one, T(0,nb-1), &ione,
                                       A(k+1,i), &ione );
            
            // Restore diagonal element, saved below during previous iteration
            *A(k+i,i-1) = ei;
        }
        
        // Generate the elementary reflector H(i) to annihilate A(k+i+1:n-1,i)
        lapackf77_dlarfg( &n_k_i_1,
                          A(k+i+1,i),
                          A(k+i+2,i), &ione, &tau[i] );
        // Save diagonal element and set to one, to simplify multiplying by V
        ei = *A(k+i+1,i);
        *A(k+i+1,i) = c_one;

        // compute yi = A vi = sum_g A{d} vi{d}
        nblocks = (n-1) / nb / ngpu + 1;
        for( d = 0; d < ngpu; ++d ) {
            magma_setdevice( d );
            magmablasSetKernelStream( data->streams[d] );
            
            // dV(k+i+1:n-1, i) = VA(k+i:n, i)
            magma_dsetvector_async( n_k_i_1,
                                    A(k+i+1,i), 1,
                                    dV(d, k+i+1, i), 1, data->streams[d] );
            
            // copy column of dV -> dVd, using block cyclic distribution.
            // This assumes V and Vd have been padded so that
            // a 2D matrix copy doesn't access them out-of-bounds
            gblock = k / nb;
            lblock = gblock / ngpu;
            lgid   = gblock % ngpu;
            if ( d < lgid ) {
                lblock += 1;
            }
            // treat V as (nb*ngpu) x nblock matrix, and Vd as nb x nblock matrix
            magmablas_dlacpy( MagmaFull, nb, nblocks-lblock,
                              dV (d, d*nb + lblock*nb*ngpu, i), nb*ngpu,
                              dVd(d, 0    + lblock*nb,      i), nb );
            
            // convert global indices (k) to local indices (dk)
            magma_indices_1D_bcyclic( nb, ngpu, d, k+i+1, n, &dki1, &dn );
            
            // dY(k:n, i) = dA(k:n, k+i+1:n) * dV(k+i+1:n, i)
            // skip if matrix is empty
            // each GPU copies to different temporary vector in Y,
            // which are summed in separate loop below
            if ( dn-dki1 > 0 ) {
                magma_dgemv( MagmaNoTrans, n-k, dn-dki1,
                             c_one,  dA (d, k,    dki1), ldda,
                                     dVd(d, dki1,    i), 1,
                             c_zero, dY (d, k,       i), 1 );
                
                // copy vector to host, storing in column nb+d of Y
                // as temporary space (Y has >= nb+ngpu columns)
                magma_dgetvector_async( n-k,
                                        dY(d, k, i), 1,
                                        Y(k, nb+d),  1, data->streams[d] );
            }
        }
        
        // while GPU is doing above Ag*v...
        // Compute T(0:i,i) = [ -tau T V' vi ]
        //                    [  tau         ]
        // T(0:i-1, i) = -tau VA(k+i+1:n-1, 0:i-1)' VA(k+i+1:n-1, i)
        scale = MAGMA_D_NEGATE( tau[i] );
        blasf77_dgemv( "Conj", &n_k_i_1, &i,
                       &scale,  A(k+i+1,0), &lda,
                                A(k+i+1,i), &ione,
                       &c_zero, T(0,i), &ione );
        // T(0:i-1, i) = T(0:i-1, 0:i-1) * T(0:i-1, i)
        blasf77_dtrmv( "Upper", "No trans", "Non-unit", &i,
                       T(0,0), &ldt,
                       T(0,i), &ione );
        *T(i,i) = tau[i];
        
        // apply reflectors to next column, A(i+1), on right only.
        // one axpy will be required to finish this, in the next iteration above
        if ( i > 0 && i+1 < nb ) {
            // Update next column, A(k:n,i+1), applying Q on right.
            // One axpy will be required to finish this, in the next iteration
            // above, after yi is computed.
            // This updates one more row than LAPACK does (row k),
            // making block above panel an even multiple of nb.
            // Use last column of T as workspace, w.
            magma_int_t i1 = i+1;
            
            // If real, conjugate row of V, and undo afterwards
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_dlacgv( &i1,  A(k+i1,0), &lda );
            #endif
            // w = T(0:i, 0:i+1) * VA(k+i+1, 0:i+1)'
            // T is now rectangular, so we use gemv instead of trmv as in lapack.
            blasf77_dgemv( "No trans", &i, &i1,
                           &c_one,  T(0,0), &ldt,
                                    A(k+i1,0), &lda,
                           &c_zero, T(0,nb-1), &ione );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_dlacgv( &i1,  A(k+i1,0), &lda );
            #endif
            
            // A(k:n, i+1) -= Y(k:n, 0:i) * w
            blasf77_dgemv( "No trans", &n_k, &i,
                           &c_neg_one, Y(k,0), &ldy,
                                       T(0,nb-1), &ione,
                           &c_one,     A(k,i1), &ione );
        }
        
        // yi = sum_g yi{d}
        for( d = 0; d < ngpu; ++d ) {
            magma_setdevice( d );
            magma_queue_sync( data->streams[d] );
            magma_indices_1D_bcyclic( nb, ngpu, d, k+i+1, n, &dki1, &dn );
            if ( dn-dki1 > 0 ) {
                // yi = yi + yi{d}
                blasf77_daxpy( &n_k, &c_one, Y(k,nb+d), &ione, Y(k,i), &ione );
            }
        }
    }
    // Restore diagonal element
    *A(k+nb,nb-1) = ei;
    
    // compute Y = Am V = sum_g Am{d} V{d} --- top part, Y(0:k-1,:)
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magmablasSetKernelStream( data->streams[d] );
        
        // convert global indices (k) to local indices (dk)
        magma_indices_1D_bcyclic( nb, ngpu, d, k+1, n, &dki1, &dn );
        
        // dY(0:k, :) = dA(0:k, k+i+1:n-1) * dV(k+i+1:n-1, :)
        // skip if matrix is empty
        // each GPU copies to different temporary block in Y,
        // which are summed in separate loop below
        if ( dn-dki1 > 0 ) {
            magma_dgemm( MagmaNoTrans, MagmaNoTrans, k, nb, dn-dki1,
                         c_one,  dA (d, 0,    dki1), ldda,
                                 dVd(d, dki1,    0), ldvd,
                         c_zero, dY (d, 0,       0), ldda );
            
            // copy result to host, storing in columns [nb + nb*d : nb + nb*(d+1)] of Y
            // as temporary space (Y has nb + nb*ngpu columns)
            magma_dgetmatrix_async( k, nb,
                                    dY(d, 0, 0),  ldda,
                                    Y(0,nb+nb*d), ldy, data->streams[d] );
        }
    }
    
    // Y = sum_g Y{d}
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magma_queue_sync( 0 );
        magma_indices_1D_bcyclic( nb, ngpu, d, k+1, n, &dki1, &dn );
        if ( dn-dki1 > 0 ) {
            // Y = Y + Am V
            for( i = 0; i < nb; ++i ) {
                blasf77_daxpy( &k, &c_one, Y(0,nb+nb*d+i), &ione, Y(0,i), &ione );
            }
        }
    }
    
    // copy Y and T matrices to GPUs
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magma_dsetmatrix_async( n, nb, Y, ldy, dY(d, 0, 0), ldda, data->streams[d] );
        magma_dsetmatrix_async( nb, nb, T, nb, dTi(d),      nb,   data->streams[d] );
    }

    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );
    
    return *info;
} /* magma_dlahr2 */
