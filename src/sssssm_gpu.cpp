/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Hatem Ltaief
       @author Mathieu Faverge

       @generated s Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_sssssm_gpu(char storev, magma_int_t m1, magma_int_t n1,
                 magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
                 float *dA1, magma_int_t ldda1,
                 float *dA2, magma_int_t ldda2,
                 float *dL1, magma_int_t lddl1,
                 float *dL2, magma_int_t lddl2,
                 magma_int_t *IPIV, magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    SSSSSM applies the LU factorization update from a real
    matrix formed by a lower triangular IB-by-K tile L1 on top of a
    M2-by-K tile L2 to a second real matrix formed by a M1-by-N1
    tile A1 on top of a M2-by-N2 tile A2 (N1 == N2).

    This is the right-looking Level 2.5 BLAS version of the algorithm.

    Arguments
    =========
    M1      (input) INTEGER
            The number of rows of the matrix A1.  M1 >= 0.

    N1      (input) INTEGER
            The number of columns of the matrix A1.  N1 >= 0.

    M2      (input) INTEGER
            The number of rows of the matrix A2.  M2 >= 0.

    N2      (input) INTEGER
            The number of columns of the matrix A2.  N2 >= 0.

    K       (input) INTEGER
            The number of columns of the matrix L1 and L2.  K >= 0.

    IB      (input) INTEGER
            The inner-blocking size.  IB >= 0.

    dA1     (input,output) REAL array, dimension(LDDA1, N), on gpu.
            On entry, the M1-by-N1 tile dA1.
            On exit, dA1 is updated by the application of dL (dL1 dL2).

    LDDA1   (input) INTEGER
            The leading dimension of the array dA1.  LDDA1 >= max(1,M1).

    dA2     (input,output) REAL array, dimension(LDDA2, N) , on gpu.
            On entry, the M2-by-N2 tile dA2.
            On exit, dA2 is updated by the application of dL (dL1 dL2).

    LDDA2   (input) INTEGER
            The leading dimension of the array dA2.  LDDA2 >= max(1,M2).

    dL1     (input) REAL array, dimension(LDDL1, K), on gpu.
            The inverse of the IB-by-K lower triangular tile as returned by
            STSTRF.

    LDDL1   (input) INTEGER
            The leading dimension of the array L1.  LDDL1 >= max(1,2*IB).

    dL2     (input) REAL array, dimension(LDDL2, K)
            The M2-by-K tile as returned by STSTRF.

    LDDL2   (input) INTEGER
            The leading dimension of the array L2.  LDDL2 >= max(1,M2).

    IPIV    (input) INTEGER array on the cpu.
            The pivot indices array of size K as returned by STSTRF

    =====================================================================    */

#define A1T(i,j) (dA1T + (i)*ldda1 + (j))
#define A2T(i,j) (dA2T + (i)*ldda2 + (j))
#define L1(i)    (dL1  + (i)*lddl1      )
#define L2(i,j)  (dL2  + (i)*lddl2i + (j)*lddl2j)

    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    int ip, ii, sb;
    float *dA1T, *dA2T;
    char transL;
    int lddl2i, lddl2j;

    /* Check input arguments */
    *info = 0;
    if (m1 < 0) {
        *info = -1;
    }
    else if (n1 < 0) {
        *info = -2;
    }
    else if (m2 < 0) {
        *info = -3;
    }
    else if (n2 < 0) {
        *info = -4;
    }
    else if (k < 0) {
        *info = -5;
    }
    else if (ib < 0) {
        *info = -6;
    }
    else if (ldda1 < max(1,m1)) {
        *info = -8;
    }
    else if (ldda2 < max(1,m2)) {
        *info = -10;
    }
    else if (lddl1 < max(1,ib)) {
        *info = -12;
    }
    else if (lddl2 < max(1,m2)) {
        *info = -14;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ((m1 == 0) || (n1 == 0) || (m2 == 0) || (n2 == 0) || (k == 0) || (ib == 0))
        return *info;

    if ( (storev == 'C') || (storev == 'c') ) {
        magmablas_sgetmo_in( dA1, dA1T, ldda1, m1, n1 );
        magmablas_sgetmo_in( dA2, dA2T, ldda2, m2, n2 );
        transL = MagmaTrans;
        lddl2i = 1; lddl2j = lddl2;
    } else {
        dA1T = dA1;
        dA2T = dA2;
        transL = MagmaNoTrans;
        lddl2i = lddl2; lddl2j = 1;
    }

    ip = 0;
    for( ii=0; ii<k; ii+=ib )
    {
        sb = min( k-ii, ib);

#ifndef NOSWAPBLK
        magmablas_sswapblk( 'R', n1,
                            A1T(0, 0), ldda1,
                            A2T(0, 0), ldda2,
                            ii+1, ii+ib, IPIV, 1, m1 );
#else
        {
            int im;
            for(i=0; i<ib; i++) {
                im = IPIV[ip]-1;

                if (im != (ii+i)) {
                    im = im - m1;

                    assert( (im>=0) && (im<m1) && (im<m2) );
                    magmablas_sswap( n1, A1T(ii+i, 0), 1, A2T(im, 0), 1 );
                }
                ip++;
            }
        }
#endif

#ifndef WITHOUTTRTRI
        /* Lower, Trans, because L1 is not transposed */
        magma_strmm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                     n1, sb,
                     c_one, L1( ii),    lddl1,
                            A1T(ii, 0), ldda1);
#else
        /* Lower, Trans, because L1 is not transposed */
        magma_strsm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                     n1, sb,
                     c_one, L1( ii),    lddl1,
                            A1T(ii, 0), ldda1);
#endif

        /* Second parameter is trans because L2 is not transposed */
        magma_sgemm( MagmaNoTrans, transL,
                     n2, m2, sb,
                     c_neg_one, A1T(ii, 0), ldda1,
                                L2( 0, ii), lddl2,
                     c_one,     A2T(0, 0 ), ldda2 );
    }

    if ( (storev == 'C') || (storev == 'c') ) {
        magmablas_sgetmo_out( dA1, dA1T, ldda1, m1, n1 );
        magmablas_sgetmo_out( dA2, dA2T, ldda2, m2, n2 );
    }
    return *info;
}
