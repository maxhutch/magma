/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zlaqps2_gpu.cu normal z -> s, Mon May  2 23:30:31 2016

*/

#include "magma_internal.h"
#include "commonblas_s.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/* --------------------------------------------------------------------------- */
/**
    Purpose
    -------
    SLAQPS computes a step of QR factorization with column pivoting
    of a real M-by-N matrix A by using Blas-3.  It tries to factorize
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
    nb      INTEGER
            The number of columns to factorize.

    @param[out]
    kb      INTEGER
            The number of columns actually factorized.

    @param[in,out]
    dA      REAL array, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    @param[out]
    dtau    REAL array, dimension (KB)
            The scalar factors of the elementary reflectors.

    @param[in,out]
    dvn1    REAL array, dimension (N)
            The vector with the partial column norms.

    @param[in,out]
    dvn2    REAL array, dimension (N)
            The vector with the exact column norms.

    @param[in,out]
    dauxv   REAL array, dimension (NB)
            Auxiliar vector.

    @param[in,out]
    dF      REAL array, dimension (LDDF,NB)
            Matrix F**H = L * Y**H * A.

    @param[in]
    lddf    INTEGER
            The leading dimension of the array F. LDDF >= max(1,N).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sgeqp3_aux
    ********************************************************************/
extern "C" magma_int_t
magma_slaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloat_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaFloat_ptr dtau, 
    magmaFloat_ptr dvn1, magmaFloat_ptr dvn2,
    magmaFloat_ptr dauxv,
    magmaFloat_ptr dF,  magma_int_t lddf,
    magmaFloat_ptr dlsticcs,
    magma_queue_t queue )
{
#define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
#define dF(i_, j_) (dF + (i_) + (j_)*(lddf))

    /* Constants */
    const float c_zero    = MAGMA_S_MAKE( 0.,0.);
    const float c_one     = MAGMA_S_MAKE( 1.,0.);
    const float c_neg_one = MAGMA_S_MAKE(-1.,0.);
    const magma_int_t ione = 1;
    
    /* Local variables */
    magma_int_t i__1, i__2;
    magma_int_t k, rk;
    float tauk;
    magma_int_t pvt, itemp;
    float tol3z;

    magmaFloat_ptr dAkk = dauxv;
    dauxv += nb;

    float lsticc;

    tol3z = magma_ssqrt( lapackf77_slamch("Epsilon"));

    lsticc = 0;
    k = 0;
    while( k < nb && lsticc == 0 ) {
        rk = offset + k;

        /* Determine ith pivot column and swap if necessary */
        pvt = k - 1 + magma_isamax( n-k, &dvn1[k], ione, queue );

        if (pvt != k) {
            magmablas_sswap( k+1, dF(pvt,0), lddf, dF(k,0), lddf, queue );

            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            magma_sswap( 2, &dvn1[pvt], n+offset, &dvn1[k], n+offset, queue );

            magmablas_sswap( m, dA(0,pvt), ione, dA(0, k), ione, queue );
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'.
           Optimization: multiply with beta=0; wait for vector and subtract */
        if (k > 0) {
            magmablas_sgemv_conj( m-rk, k,
                                  c_neg_one, dA(rk, 0), ldda,
                                             dF(k,  0), lddf,
                                  c_one,     dA(rk, k), ione, queue );
        }

        /*  Generate elementary reflector H(k). */
        magma_slarfg_gpu( m-rk, dA(rk, k), dA(rk + 1, k), &dtau[k], &dvn1[k], &dAkk[k], queue );
        magma_ssetvector( 1, &c_one,   1, dA(rk, k), 1, queue );

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1 || k > 0 ) {
            magma_sgetvector( 1, &dtau[k], 1, &tauk, 1, queue );
        }
        if (k < n-1) {
            magma_sgemv( MagmaConjTrans, m-rk, n-k-1,
                         tauk,   dA( rk,  k+1 ), ldda,
                                 dA( rk,  k   ), 1,
                         c_zero, dF( k+1, k   ), 1, queue );
        }

        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). 
           F(1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K)
                    := tau(K)(A(RK:M,K+1:N)' - F(1:N,1:K-1)*A(RK:M,1:K-1)') A(RK:M,K)  
           so, F is (updated A)*V */
        if (k > 0) {
            /*z__1 = MAGMA_S_NEGATE( tauk );
            magma_sgemv( MagmaConjTrans, m-rk, k,
                         z__1,   dA(rk, 0), ldda,
                                 dA(rk, k), ione,
                         c_zero, dauxv, ione, queue ); */

            magma_sgemv_kernel3
                <<< k, BLOCK_SIZE, 0, queue->cuda_stream() >>>
                (m-rk, dA(rk, 0), ldda, dA(rk, k), dauxv, dtau+k);

            /* I think we only need stricly lower-triangular part */
            magma_sgemv( MagmaNoTrans, n-k-1, k,
                         c_one, dF(k+1,0), lddf,
                                dauxv,     ione,
                         c_one, dF(k+1,k), ione, queue );
        }

       /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            /* left-looking update of rows,                     *
             * since F=A**H v with original A, so no right-looking */
            magma_sgemm( MagmaNoTrans, MagmaConjTrans, ione, i__1, i__2,
                         c_neg_one, dA(rk, 0  ), ldda,
                                    dF(k+1,0  ), lddf,
                         c_one,     dA(rk, k+1), ldda, queue ); 
        }

        /* Update partial column norms. */
        if (rk < min(m, n+offset)-1) {
            magmablas_snrm2_row_check_adjust( n-k-1, tol3z, &dvn1[k+1], 
                                               &dvn2[k+1], dA(rk,k+1), ldda, dlsticcs, queue ); 
            
            magma_sgetvector( 1, &dlsticcs[0], 1, &lsticc, 1, queue );
        }

        //*dA(rk, k) = Akk;
        //magma_ssetvector( 1, &Akk, 1, dA(rk, k), 1, queue );
        //magmablas_slacpy( MagmaFull, 1, 1, dAkk, 1, dA(rk, k), 1, queue );

        ++k;
    }
    // restore the diagonals
    magma_scopymatrix( 1, k, dAkk, 1, dA(offset, 0), ldda+1, queue );

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

        magma_sgemm( MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb,
                     c_neg_one, dA(rk+1, 0  ), ldda,
                                dF(*kb,  0  ), lddf,
                     c_one,     dA(rk+1, *kb), ldda, queue );
    }

    /* Recomputation of difficult columns. */
    if ( lsticc > 0 ) {
        // printf( " -- recompute dnorms --\n" );
        magmablas_snrm2_check( m-rk-1, n-*kb, dA(rk+1,*kb), ldda,
                               &dvn1[*kb], dlsticcs, queue );
        magma_scopymatrix( n-*kb, 1, &dvn1[*kb], n, &dvn2[*kb], n, queue );
    }
    
    return MAGMA_SUCCESS;
} /* magma_slaqps2_q */
