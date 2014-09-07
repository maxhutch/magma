/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c

*/

#include "common_magma.h"
#include "commonblas_z.h"

#define PRECISION_z

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/* --------------------------------------------------------------------------- */
/**
    Purpose
    -------
    ZLAQPS computes a step of QR factorization with column pivoting
    of a complex M-by-N matrix A by using Blas-3.  It tries to factorize
    NB columns from A starting from the row OFFSET+1, and updates all
    of the matrix with Blas-3 xGEMM.

    In some cases, due to catastrophic cancellations, it cannot
    factorize NB columns.  Hence, the actual number of factorized
    columns is returned in KB.

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. N >= 0

    @param[in]
    offset  INTEGER
            The number of rows of A that have been factorized in
            previous steps.

    @param[in]
    NB      INTEGER
            The number of columns to factorize.

    @param[out]
    kb      INTEGER
            The number of columns actually factorized.

    @param[in,out]
    A       COMPLEX*16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    @param[out]
    tau     COMPLEX*16 array, dimension (KB)
            The scalar factors of the elementary reflectors.

    @param[in,out]
    VN1     DOUBLE PRECISION array, dimension (N)
            The vector with the partial column norms.

    @param[in,out]
    VN2     DOUBLE PRECISION array, dimension (N)
            The vector with the exact column norms.

    @param[in,out]
    AUXV    COMPLEX*16 array, dimension (NB)
            Auxiliar vector.

    @param[in,out]
    F       COMPLEX*16 array, dimension (LDF,NB)
            Matrix F' = L*Y'*A.

    @param[in]
    ldf     INTEGER
            The leading dimension of the array F. LDF >= max(1,N).

    @ingroup magma_zgeqp3_aux
    ********************************************************************/
extern "C" magma_int_t
magma_zlaqps2_gpu(magma_int_t m, magma_int_t n, magma_int_t offset,
             magma_int_t nb, magma_int_t *kb,
             magmaDoubleComplex *A,  magma_int_t lda,
             magma_int_t *jpvt, magmaDoubleComplex *tau, 
             double *vn1, double *vn2,
             magmaDoubleComplex *auxv,
             magmaDoubleComplex *F,  magma_int_t ldf)
{
#define  A(i, j) (A  + (i) + (j)*(lda ))
#define  F(i, j) (F  + (i) + (j)*(ldf ))

    magmaDoubleComplex c_zero    = MAGMA_Z_MAKE( 0.,0.);
    magmaDoubleComplex c_one     = MAGMA_Z_MAKE( 1.,0.);
    magmaDoubleComplex c_neg_one = MAGMA_Z_MAKE(-1.,0.);
    magma_int_t ione = 1;
    
    magma_int_t i__1, i__2;
    
    magma_int_t k, rk;
    magmaDoubleComplex tauk;
    magma_int_t pvt, itemp;
    double tol3z;

    magmaDoubleComplex *dAkk = auxv;
    auxv+=nb;

    double lsticc, *lsticcs;
    magma_dmalloc( &lsticcs, 1+256*(n+255)/256 );

    tol3z = magma_dsqrt( lapackf77_dlamch("Epsilon"));

    lsticc = 0;
    k = 0;
    while( k < nb && lsticc == 0 ) {
        rk = offset + k;

        /* Determine ith pivot column and swap if necessary */
        pvt = k - 1 + magma_idamax( n-k, &vn1[k], ione );

        if (pvt != k) {
             magmablas_zswap( k+1, F(pvt,0), ldf, F(k,0), ldf);

            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            #if (defined(PRECISION_d) || defined(PRECISION_z))
                //magma_dswap( 1, &vn1[pvt], 1, &vn1[k], 1 );
                //magma_dswap( 1, &vn2[pvt], 1, &vn2[k], 1 );
                magma_dswap( 2, &vn1[pvt], n+offset, &vn1[k], n+offset);
            #else
                //magma_sswap( 1, &vn1[pvt], 1, &vn1[k], 1 );
                //magma_sswap( 1, &vn2[pvt], 1, &vn2[k], 1 );
                magma_sswap(2, &vn1[pvt], n+offset, &vn1[k], n+offset);
            #endif

            magmablas_zswap( m, A(0,pvt), ione, A(0, k), ione );
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'.
           Optimization: multiply with beta=0; wait for vector and subtract */
        if (k > 0) {
            magmablas_zgemv_conjv( m-rk, k,
                                   c_neg_one, A(rk, 0), lda,
                                              F(k,  0), ldf,
                                   c_one,     A(rk, k), ione );
        }

        /*  Generate elementary reflector H(k). */
        magma_zlarfg_gpu(m-rk, A(rk, k), A(rk + 1, k), &tau[k], &vn1[k], &dAkk[k]);
        magma_zsetvector( 1, &c_one,   1, A(rk, k), 1 );

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1 || k > 0 ) magma_zgetvector( 1, &tau[k], 1, &tauk, 1 );
        if (k < n-1) {
            magma_zgemv( MagmaConjTrans, m-rk, n-k-1,
                     tauk,   A( rk,  k+1 ), lda,
                             A( rk,  k   ), 1,
                     c_zero, F( k+1, k   ), 1 );
        }

        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). 
           F(1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K)
                    := tau(K)(A(RK:M,K+1:N)' - F(1:N,1:K-1)*A(RK:M,1:K-1)') A(RK:M,K)  
           so, F is (updated A)*V */
        if (k > 0) {
            /*z__1 = MAGMA_Z_NEGATE( tauk );
            magma_zgemv( MagmaConjTrans, m-rk, k,
                         z__1,   A(rk, 0), lda,
                                 A(rk, k), ione,
                         c_zero, auxv, ione );*/

            magma_zgemv_kernel3<<< k, BLOCK_SIZE, 0, magma_stream >>>(m-rk, A(rk, 0), lda,
                                                                      A(rk, k), auxv, tau+k);

            /* I think we only need stricly lower-triangular part */
            magma_zgemv( MagmaNoTrans, n-k-1, k,
                         c_one, F(k+1,0), ldf,
                                auxv,     ione,
                         c_one, F(k+1,k), ione );
        }

       /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            /* left-looking update of rows,                     *
             * since F=A'v with original A, so no right-looking */
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, ione, i__1, i__2,
                         c_neg_one, A(rk, 0  ), lda,
                                    F(k+1,0  ), ldf,
                         c_one,     A(rk, k+1), lda ); 
        }

        /* Update partial column norms. */
        if (rk < min(m, n+offset)-1){
           magmablas_dznrm2_row_check_adjust(n-k-1, tol3z, &vn1[k+1], 
                                             &vn2[k+1], A(rk,k+1), lda, lsticcs); 

           #if defined(PRECISION_d) || defined(PRECISION_z)
               magma_dgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
           #else
               magma_sgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
           #endif
        }

        //*A(rk, k) = Akk;
        //magma_zsetvector( 1, &Akk, 1, A(rk, k), 1 );
        //magmablas_zlacpy(MagmaUpperLower, 1, 1, dAkk, 1, A(rk, k), 1);

        ++k;
    }
    // restore the diagonals
    magma_zcopymatrix( 1, k, dAkk, 1, A(offset, 0), lda+1 );

    // leave k as the last column done
    --k;
    *kb = k + 1;
    rk = offset + *kb - 1;

    /* Apply the block reflector to the rest of the matrix:
       A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - 
                                  A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'  */
    if (*kb < min(n, m - offset)) {
        i__1 = m - rk - 1;
        i__2 = n - *kb;

        magma_zgemm( MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb,
                     c_neg_one, A(rk+1, 0  ), lda,
                                F(*kb,  0  ), ldf,
                     c_one,     A(rk+1, *kb), lda );
    }

    /* Recomputation of difficult columns. */
    if( lsticc > 0 ) {
        // printf( " -- recompute dnorms --\n" );
        magmablas_dznrm2_check(m-rk-1, n-*kb, A(rk+1,*kb), lda,
                               &vn1[*kb], lsticcs);
#if defined(PRECISION_d) || defined(PRECISION_z)
        magma_dcopymatrix( n-*kb, 1, &vn1[*kb], n, &vn2[*kb], n);
#else   
        magma_scopymatrix( n-*kb, 1, &vn1[*kb], n, &vn2[*kb], n);
#endif  
    }
    magma_free(lsticcs);
    
    return MAGMA_SUCCESS;
} /* magma_zlaqps */
