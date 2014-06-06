/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Hatem Ltaief
       @author Mathieu Faverge

       @generated d Tue Dec 17 13:18:36 2013

*/
#ifdef MAGMA_WITH_PLASMA

#include <plasma.h>
#include <core_blas.h>
#include "common_magma.h"


extern "C" magma_int_t
magma_dtstrf_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                  double *hU, magma_int_t ldhu, double *dU, magma_int_t lddu,
                  double *hA, magma_int_t ldha, double *dA, magma_int_t ldda,
                  double *hL, magma_int_t ldhl, double *dL, magma_int_t lddl,
                  magma_int_t *ipiv,
                  double *hwork, magma_int_t ldhwork, double *dwork, magma_int_t lddwork,
                  magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    DSSSSM applies the LU factorization update from a real
    matrix formed by a lower triangular IB-by-K tile L1 on top of a
    M2-by-K tile L2 to a second real matrix formed by a M1-by-N1
    tile A1 on top of a M2-by-N2 tile A2 (N1 == N2).

    This is the right-looking Level 2.5 BLAS version of the algorithm.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    IB      (input) INTEGER
            The inner-blocking size.  IB >= 0.

    NB      (input) INTEGER
            The blocking size.  NB >= 0.

    hU      (input,output) DOUBLE_PRECISION array, dimension(LDHU, N), on cpu.
            On entry, the NB-by-N upper triangular tile hU.
            On exit, the content is incomplete. Shouldn't be used.

    LDHU    (input) INTEGER
            The leading dimension of the array hU.  LDHU >= max(1,NB).

    dU      (input,output) DOUBLE_PRECISION array, dimension(LDDU, N), on gpu.
            On entry, the NB-by-N upper triangular tile dU identical to hU.
            On exit, the new factor U from the factorization.

    LDDU    (input) INTEGER
            The leading dimension of the array dU.  LDDU >= max(1,NB).

    hA      (input,output) DOUBLE_PRECISION array, dimension(LDHA, N), on cpu.
            On entry, only the M-by-IB first panel needs to be identical to dA(1..M, 1..IB).
            On exit, the content is incomplete. Shouldn't be used.

    LDHA    (input) INTEGER
            The leading dimension of the array hA.  LDHA >= max(1,M).

    dA      (input,output) DOUBLE_PRECISION array, dimension(LDDA, N) , on gpu.
            On entry, the M-by-N tile to be factored.
            On exit, the factor L from the factorization

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    hL      (output) DOUBLE_PRECISION array, dimension(LDHL, K), on vpu.
            On exit, contains in the upper part the IB-by-K lower triangular tile,
            and in the lower part IB-by-K the inverse of the top part.

    LDHL    (input) INTEGER
            The leading dimension of the array hL.  LDHL >= max(1,2*IB).

    dL      (output) DOUBLE_PRECISION array, dimension(LDDL, K), on gpu.
            On exit, contains in the upper part the IB-by-K lower triangular tile,
            and in the lower part IB-by-K the inverse of the top part.

    LDDL    (input) INTEGER
            The leading dimension of the array dL.  LDDL >= max(1,2*IB).

    hWORK   (output) DOUBLE_PRECISION array, dimension(LDHWORK, 2*IB), on cpu.
            Workspace.

    LDHWORK (input) INTEGER
            The leading dimension of the array hWORK.  LDHWORK >= max(NB, 1).

    dWORK   (output) DOUBLE_PRECISION array, dimension(LDDWORK, 2*IB), on gpu.
            Workspace.

    LDDWORK (input) INTEGER
            The leading dimension of the array dWORK.  LDDWORK >= max(NB, 1).

    IPIV    (output) INTEGER array on the cpu.
            The pivot indices array of size K as returned by DTSTRF

    INFO    (output) INTEGER
            - PLASMA_SUCCESS successful exit
            - < 0 if INFO = -k, the k-th argument had an illegal value
            - > 0 if INFO = k, U(k,k) is exactly zero. The factorization
                has been completed, but the factor U is exactly
                singular, and division by zero will occur if it is used
                to solve a system of equations.

    =====================================================================    */

#define UT(i,j) (dUT + (i)*ib*lddu + (j)*ib )
#define AT(i,j) (dAT + (i)*ib*ldda + (j)*ib )
#define L(i)    (dL  + (i)*ib*lddl          )
#define L2(i)   (dL2 + (i)*ib*lddl          )
#define hU(i,j) (hU  + (j)*ib*ldhu + (i)*ib )
#define hA(i,j) (hA  + (j)*ib*ldha + (i)*ib )
#define hL(i)   (hL  + (i)*ib*ldhl          )
#define hL2(i)  (hL2 + (i)*ib*ldhl          )

    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    int iinfo = 0;
    int maxm, mindim;
    int i, j, im, s, ip, ii, sb, p = 1;
    double *dAT, *dUT;
    double *dAp, *dUp;
#ifndef WITHOUTTRTRI
    double *dL2 = dL + ib;
    double *hL2 = hL + ib;
    p = 2;
#endif

    /* Check input arguments */
    *info = 0;
    if (m < 0) {
        *info = -1;
    }
    else if (n < 0) {
        *info = -2;
    }
    else if (ib < 0) {
        *info = -3;
    }
    else if ((lddu < max(1,m)) && (m > 0)) {
        *info = -6;
    }
    else if ((ldda < max(1,m)) && (m > 0)) {
        *info = -8;
    }
    else if ((lddl < max(1,ib)) && (ib > 0)) {
        *info = -10;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* quick return */
    if ((m == 0) || (n == 0) || (ib == 0))
        return *info;

    ip = 0;

    /* Function Body */
    mindim = min(m, n);
    s      = mindim / ib;

    if ( ib >= mindim ) {
        /* Use CPU code. */
        CORE_dtstrf(m, n, ib, nb,
                    (double*)hU, ldhu,
                    (double*)hA, ldha,
                    (double*)hL, ldhl,
                    ipiv,
                    (double*)hwork, ldhwork,
                    info);

#ifndef WITHOUTTRTRI
        CORE_dlacpy( PlasmaUpperLower, mindim, mindim,
                     (double*)hL, ldhl,
                     (double*)hL2, ldhl );
        CORE_dtrtri( PlasmaLower, PlasmaUnit, mindim,
                     (double*)hL2, ldhl, info );
        if (*info != 0 ) {
            fprintf(stderr, "ERROR, trtri returned with info = %d\n", *info);
        }
#endif

        if ( (storev == 'R') || (storev == 'r') ) {
            magma_dsetmatrix( m, n, hU, ldhu, dwork, lddwork );
            magmablas_dtranspose( dU, lddu, dwork, lddwork, m, n );

            magma_dsetmatrix( m, n, hA, ldha, dwork, lddwork );
            magmablas_dtranspose( dA, ldda, dwork, lddwork, m, n );
        } else {
            magma_dsetmatrix( m, n, hU, ldhu, dU, lddu );
            magma_dsetmatrix( m, n, hA, ldha, dA, ldda );
        }
        magma_dsetmatrix( p*ib, n, hL, ldhl, dL, lddl );
            
    }
    else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31)/32)*32;

        if ( (storev == 'C') || (storev == 'c') ) {
            magmablas_dgetmo_in( dU, dUT, lddu, m,  n );
            magmablas_dgetmo_in( dA, dAT, ldda, m,  n );
        } else {
            dUT = dU; dAT = dA;
        }
        dAp = dwork;
        dUp = dAp + ib*lddwork;

        ip = 0;
        for( i=0; i<s; i++ )
        {
            ii = i * ib;
            sb = min(mindim-ii, ib);
            
            if ( i>0 ){
                // download i-th panel
                magmablas_dtranspose( dUp, lddu, UT(0, i), lddu, sb, ii );
                magmablas_dtranspose( dAp, ldda, AT(0, i), ldda, sb, m  );
                
                magma_dgetmatrix( ii, sb, dUp, lddu, hU(0, i), ldhu );
                magma_dgetmatrix( m, sb, dAp, ldda, hA(0, i), ldha );
                
                // make sure that gpu queue is empty
                //magma_device_sync();
                
#ifndef WITHOUTTRTRI
                magma_dtrmm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                             n-(ii+sb), ib,
                             c_one, L2(i-1),      lddl,
                                    UT(i-1, i+1), lddu);
#else
                magma_dtrsm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                             n-(ii+sb), ib,
                             c_one, L(i-1),       lddl,
                                    UT(i-1, i+1), lddu);
#endif
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(ii+sb), m, ib,
                             c_neg_one, UT(i-1, i+1), lddu,
                                        AT(0,   i-1), ldda,
                             c_one,     AT(0,   i+1), ldda );
            }

            // do the cpu part
            CORE_dtstrf(m, sb, ib, nb,
                        (double*)hU(i, i), ldhu,
                        (double*)hA(0, i), ldha,
                        (double*)hL(i),    ldhl,
                        ipiv+ii,
                        (double*)hwork, ldhwork,
                        info);

            if ( (*info == 0) && (iinfo > 0) )
                *info = iinfo + ii;
            
            // Need to swap betw U and A
#ifndef NOSWAPBLK
            magmablas_dswapblk( 'R', n-(ii+sb),
                                UT(i, i+1), lddu,
                                AT(0, i+1), ldda,
                                1, sb, ipiv+ii, 1, nb );

            for(j=0; j<ib; j++) {
                im = ipiv[ip]-1;
                if ( im == j ) {
                    ipiv[ip] += ii;
                }
                ip++;
            }
#else
            for(j=0; j<ib; j++) {
                im = ipiv[ip]-1;
                if ( im != (j) ) {
                    im = im - nb;
                    assert( (im>=0) && (im<m) );
                    magmablas_dswap( n-(ii+sb), UT(i, i+1)+j*lddu, 1, AT(0, i+1)+im*ldda, 1 );
                } else {
                    ipiv[ip] += ii;
                }
                ip++;
            }
#endif

#ifndef WITHOUTTRTRI
            CORE_dlacpy( PlasmaUpperLower, sb, sb,
                         (double*)hL(i), ldhl,
                         (double*)hL2(i), ldhl );
            CORE_dtrtri( PlasmaLower, PlasmaUnit, sb,
                         (double*)hL2(i), ldhl, info );
            if (*info != 0 ) {
                fprintf(stderr, "ERROR, trtri returned with info = %d\n", *info);
            }
#endif
            // upload i-th panel
            magma_dsetmatrix( sb, sb, hU(i, i), ldhu, dUp, lddu );
            magma_dsetmatrix( m, sb, hA(0, i), ldha, dAp, ldda );
            magma_dsetmatrix( p*ib, sb, hL(i), ldhl, L(i), lddl );
            magmablas_dtranspose( UT(i, i), lddu, dUp, lddu, sb, sb);
            magmablas_dtranspose( AT(0, i), ldda, dAp, ldda, m,  sb);
            
            // make sure that gpu queue is empty
            //magma_device_sync();
            
            // do the small non-parallel computations
            if ( s > (i+1) ) {
#ifndef WITHOUTTRTRI
                magma_dtrmm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                             sb, sb,
                             c_one, L2(i),      lddl,
                                    UT(i, i+1), lddu);
#else
                magma_dtrsm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                             sb, sb,
                             c_one, L(i),      lddl,
                                    UT(i, i+1), lddu);
#endif
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             sb, m, sb,
                             c_neg_one, UT(i, i+1), lddu,
                                        AT(0, i  ), ldda,
                             c_one,     AT(0, i+1), ldda );
            }
            else {
#ifndef WITHOUTTRTRI
                magma_dtrmm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                             n-mindim, sb,
                             c_one, L2(i),      lddl,
                                    UT(i, i+1), lddu);
#else
                magma_dtrsm( MagmaRight, MagmaLower, MagmaTrans, MagmaUnit,
                             n-mindim, sb,
                             c_one, L(i),      lddl,
                                    UT(i, i+1), lddu);
#endif
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             n-mindim, m, sb,
                             c_neg_one, UT(i, i+1), lddu,
                                        AT(0, i  ), ldda,
                             c_one,     AT(0, i+1), ldda );
            }
        }

        if ( (storev == 'C') || (storev == 'c') ) {
            magmablas_dgetmo_out( dU, dUT, lddu, m,  n );
            magmablas_dgetmo_out( dA, dAT, ldda, m,  n );
        }
    }
    return *info;
}

#endif /* MAGMA_WITH_PLASMA */
