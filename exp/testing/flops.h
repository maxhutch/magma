/**
 *
 * @file flops.h
 *
 *  File provided by Univ. of Tennessee,
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date January 2015
 *
 **/
/*
 * This file provide the flops formula for all Level 3 BLAS and some
 * Lapack routines.  Each macro uses the same size parameters as the
 * function associated and provide one formula for additions and one
 * for multiplications. Example to use these macros:
 *
 *    FLOPS_ZGEMM( m, n, k )
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */
#ifndef _FLOPS_H_
#define _FLOPS_H_

/************************************************************************
 *           Generic formula coming from LAWN 41
 ***********************************************************************/

/*
 * Level 2 BLAS 
 */  
#define FMULS_GEMV(__m, __n) ((__m) * (__n) + 2. * (__m))
#define FADDS_GEMV(__m, __n) ((__m) * (__n)           )

#define FMULS_SYMV(__n) FMULS_GEMV( (__n), (__n) )
#define FADDS_SYMV(__n) FADDS_GEMV( (__n), (__n) )
#define FMULS_HEMV FMULS_SYMV
#define FADDS_HEMV FADDS_SYMV

/*
 * Level 3 BLAS 
 */
#define FMULS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))
#define FADDS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))

#define FMULS_SYMM(__side, __m, __n) ( ( (__side) == PlasmaLeft ) ? FMULS_GEMM((__m), (__m), (__n)) : FMULS_GEMM((__m), (__n), (__n)) )
#define FADDS_SYMM(__side, __m, __n) ( ( (__side) == PlasmaLeft ) ? FADDS_GEMM((__m), (__m), (__n)) : FADDS_GEMM((__m), (__n), (__n)) )
#define FMULS_HEMM FMULS_SYMM
#define FADDS_HEMM FADDS_SYMM

#define FMULS_SYRK(__k, __n) (0.5 * (__k) * (__n) * ((__n)+1))
#define FADDS_SYRK(__k, __n) (0.5 * (__k) * (__n) * ((__n)+1))
#define FMULS_HERK FMULS_SYRK
#define FADDS_HERK FADDS_SYRK

#define FMULS_SYR2K(__k, __n) ((__k) * (__n) * (__n)        )
#define FADDS_SYR2K(__k, __n) ((__k) * (__n) * (__n) + (__n))
#define FMULS_HER2K FMULS_SYR2K
#define FADDS_HER2K FADDS_SYR2K

#define FMULS_TRMM_2(__m, __n) (0.5 * (__n) * (__m) * ((__m)+1))
#define FADDS_TRMM_2(__m, __n) (0.5 * (__n) * (__m) * ((__m)-1))


#define FMULS_TRMM(__side, __m, __n) ( ( (__side) == PlasmaLeft ) ? FMULS_TRMM_2((__m), (__n)) : FMULS_TRMM_2((__n), (__m)) )
#define FADDS_TRMM(__side, __m, __n) ( ( (__side) == PlasmaLeft ) ? FADDS_TRMM_2((__m), (__n)) : FADDS_TRMM_2((__n), (__m)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM

/*
 * Lapack
 */
#define FMULS_GETRF(__m, __n) ( ((__m) < (__n)) ? (0.5 * (__m) * ((__m) * ((__n) - (1./3.) * (__m) - 1. ) + (__n)) + (2. / 3.) * (__m)) \
                                :                 (0.5 * (__n) * ((__n) * ((__m) - (1./3.) * (__n) - 1. ) + (__m)) + (2. / 3.) * (__n)) )
#define FADDS_GETRF(__m, __n) ( ((__m) < (__n)) ? (0.5 * (__m) * ((__m) * ((__n) - (1./3.) * (__m)      ) - (__n)) + (1. / 6.) * (__m)) \
                                :                 (0.5 * (__n) * ((__n) * ((__m) - (1./3.) * (__n)      ) - (__m)) + (1. / 6.) * (__n)) )

#define FMULS_GETRI(__n) ( (__n) * ((5. / 6.) + (__n) * ((2. / 3.) * (__n) + 0.5)) )
#define FADDS_GETRI(__n) ( (__n) * ((5. / 6.) + (__n) * ((2. / 3.) * (__n) - 1.5)) )

#define FMULS_GETRS(__n, __nrhs) ((__nrhs) * (__n) *  (__n)      )
#define FADDS_GETRS(__n, __nrhs) ((__nrhs) * (__n) * ((__n) - 1 ))

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))

#define FMULS_POTRI(__n) ( (__n) * ((2. / 3.) + (__n) * ((1. / 3.) * (__n) + 1. )) )
#define FADDS_POTRI(__n) ( (__n) * ((1. / 6.) + (__n) * ((1. / 3.) * (__n) - 0.5)) )

#define FMULS_POTRS(__n, __nrhs) ((__nrhs) * (__n) * ((__n) + 1 ))
#define FADDS_POTRS(__n, __nrhs) ((__nrhs) * (__n) * ((__n) - 1 ))

//SPBTRF
//SPBTRS
//SSYTRF
//SSYTRI
//SSYTRS

#define FMULS_GEQRF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * (  0.5-(1./3.) * (__n) + (__m)) +    (__m) + 23. / 6.)) \
                               :                 ((__m) * ((__m) * ( -0.5-(1./3.) * (__m) + (__n)) + 2.*(__n) + 23. / 6.)) )
#define FADDS_GEQRF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * (  0.5-(1./3.) * (__n) + (__m))            +  5. / 6.)) \
                               :                 ((__m) * ((__m) * ( -0.5-(1./3.) * (__m) + (__n)) +    (__n) +  5. / 6.)) )

#define FMULS_GEQLF(__m, __n) FMULS_GEQRF(__m, __n)
#define FADDS_GEQLF(__m, __n) FADDS_GEQRF(__m, __n)

#define FMULS_GERQF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * (  0.5-(1./3.) * (__n) + (__m)) +    (__m) + 29. / 6.)) \
                               :                 ((__m) * ((__m) * ( -0.5-(1./3.) * (__m) + (__n)) + 2.*(__n) + 29. / 6.)) )
#define FADDS_GERQF(__m, __n) (((__m) > (__n)) ? ((__n) * ((__n) * ( -0.5-(1./3.) * (__n) + (__m)) +    (__m) +  5. / 6.)) \
                               :                 ((__m) * ((__m) * (  0.5-(1./3.) * (__m) + (__n)) +          +  5. / 6.)) )

#define FMULS_GELQF(__m, __n) FMULS_GERQF(__m, __n)
#define FADDS_GELQF(__m, __n) FADDS_GERQF(__m, __n)

#define FMULS_UNGQR(__m, __n, __k) ((__k) * (2.* (__m) * (__n) +    2. * (__n) - 5./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n)) - 1.)))
#define FADDS_UNGQR(__m, __n, __k) ((__k) * (2.* (__m) * (__n) + (__n) - (__m) + 1./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n))     )))
#define FMULS_UNGQL FMULS_UNGQR
#define FMULS_ORGQR FMULS_UNGQR
#define FMULS_ORGQL FMULS_UNGQR
#define FADDS_UNGQL FADDS_UNGQR
#define FADDS_ORGQR FADDS_UNGQR
#define FADDS_ORGQL FADDS_UNGQR

#define FMULS_UNGRQ(__m, __n, __k) ((__k) * (2.* (__m) * (__n) + (__m) + (__n) - 2./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n)) - 1.)))
#define FADDS_UNGRQ(__m, __n, __k) ((__k) * (2.* (__m) * (__n) + (__m) - (__n) + 1./3. + (__k) * ( 2./3. * (__k) - ((__m) + (__n))     )))
#define FMULS_UNGLQ FMULS_UNGRQ
#define FMULS_ORGRQ FMULS_UNGRQ
#define FMULS_ORGLQ FMULS_UNGRQ
#define FADDS_UNGLQ FADDS_UNGRQ
#define FADDS_ORGRQ FADDS_UNGRQ
#define FADDS_ORGLQ FADDS_UNGRQ

#define FMULS_GEQRS(__m, __n, __nrhs) ((__nrhs) * ((__n) * ( 2.* (__m) - 0.5 * (__n) + 2.5)))
#define FADDS_GEQRS(__m, __n, __nrhs) ((__nrhs) * ((__n) * ( 2.* (__m) - 0.5 * (__n) + 0.5)))

//UNMQR, UNMLQ, UNMQL, UNMRQ (Left)
//UNMQR, UNMLQ, UNMQL, UNMRQ (Right)

#define FMULS_TRTRI(__n) ((__n) * ((__n) * ( 1./6. * (__n) + 0.5 ) + 1./3.))
#define FADDS_TRTRI(__n) ((__n) * ((__n) * ( 1./6. * (__n) - 0.5 ) + 1./3.))

#define FMULS_GEHRD(__n) ( (__n) * ((__n) * (5./3. *(__n) + 0.5) - 7./6.) - 13. )
#define FADDS_GEHRD(__n) ( (__n) * ((__n) * (5./3. *(__n) - 1. ) - 2./3.) -  8. )

#define FMULS_SYTRD(__n) ( (__n) *  ( (__n) * ( 2./3. * (__n) + 2.5 ) - 1./6. ) - 15.)
#define FADDS_SYTRD(__n) ( (__n) *  ( (__n) * ( 2./3. * (__n) + 1.  ) - 8./3. ) -  4.)
#define FMULS_HETRD FMULS_SYTRD
#define FADDS_HETRD FADDS_SYTRD

#define FMULS_GEBRD(__m, __n) ( ((__m) >= (__n)) ? ((__n) * ((__n) * (2. * (__m) - 2./3. * (__n) + 2. )         + 20./3.)) \
                                :                  ((__m) * ((__m) * (2. * (__n) - 2./3. * (__m) + 2. )         + 20./3.)) )
#define FADDS_GEBRD(__m, __n) ( ((__m) >= (__n)) ? ((__n) * ((__n) * (2. * (__m) - 2./3. * (__n) + 1. ) - (__m) +  5./3.)) \
                                :                  ((__m) * ((__m) * (2. * (__n) - 2./3. * (__m) + 1. ) - (__n) +  5./3.)) )


/*******************************************************************************
 *               Users functions
 ******************************************************************************/

/*
 * Level 2 BLAS 
 */  
#define FLOPS_ZGEMV(__m, __n) (6. * FMULS_GEMV((double)(__m), (double)(__n)) + 2.0 * FADDS_GEMV((double)(__m), (double)(__n)) )
#define FLOPS_CGEMV(__m, __n) (6. * FMULS_GEMV((double)(__m), (double)(__n)) + 2.0 * FADDS_GEMV((double)(__m), (double)(__n)) )
#define FLOPS_DGEMV(__m, __n) (     FMULS_GEMV((double)(__m), (double)(__n)) +       FADDS_GEMV((double)(__m), (double)(__n)) )
#define FLOPS_SGEMV(__m, __n) (     FMULS_GEMV((double)(__m), (double)(__n)) +       FADDS_GEMV((double)(__m), (double)(__n)) )
 
#define FLOPS_ZHEMV(__n) (6. * FMULS_HEMV((double)(__n)) + 2.0 * FADDS_HEMV((double)(__n)) )
#define FLOPS_CHEMV(__n) (6. * FMULS_HEMV((double)(__n)) + 2.0 * FADDS_HEMV((double)(__n)) )

#define FLOPS_ZSYMV(__n) (6. * FMULS_SYMV((double)(__n)) + 2.0 * FADDS_SYMV((double)(__n)) )
#define FLOPS_CSYMV(__n) (6. * FMULS_SYMV((double)(__n)) + 2.0 * FADDS_SYMV((double)(__n)) )
#define FLOPS_DSYMV(__n) (     FMULS_SYMV((double)(__n)) +       FADDS_SYMV((double)(__n)) )
#define FLOPS_SSYMV(__n) (     FMULS_SYMV((double)(__n)) +       FADDS_SYMV((double)(__n)) )
 
/*
 * Level 3 BLAS 
 */
#define FLOPS_ZGEMM(__m, __n, __k) (6. * FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CGEMM(__m, __n, __k) (6. * FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DGEMM(__m, __n, __k) (     FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) +       FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SGEMM(__m, __n, __k) (     FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) +       FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZHEMM(__side, __m, __n) (6. * FMULS_HEMM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_HEMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_CHEMM(__side, __m, __n) (6. * FMULS_HEMM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_HEMM(__side, (double)(__m), (double)(__n)) )

#define FLOPS_ZSYMM(__side, __m, __n) (6. * FMULS_SYMM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_SYMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_CSYMM(__side, __m, __n) (6. * FMULS_SYMM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_SYMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_DSYMM(__side, __m, __n) (     FMULS_SYMM(__side, (double)(__m), (double)(__n)) +       FADDS_SYMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_SSYMM(__side, __m, __n) (     FMULS_SYMM(__side, (double)(__m), (double)(__n)) +       FADDS_SYMM(__side, (double)(__m), (double)(__n)) )
 
#define FLOPS_ZHERK(__k, __n) (6. * FMULS_HERK((double)(__k), (double)(__n)) + 2.0 * FADDS_HERK((double)(__k), (double)(__n)) )
#define FLOPS_CHERK(__k, __n) (6. * FMULS_HERK((double)(__k), (double)(__n)) + 2.0 * FADDS_HERK((double)(__k), (double)(__n)) )

#define FLOPS_ZSYRK(__k, __n) (6. * FMULS_SYRK((double)(__k), (double)(__n)) + 2.0 * FADDS_SYRK((double)(__k), (double)(__n)) )
#define FLOPS_CSYRK(__k, __n) (6. * FMULS_SYRK((double)(__k), (double)(__n)) + 2.0 * FADDS_SYRK((double)(__k), (double)(__n)) )
#define FLOPS_DSYRK(__k, __n) (     FMULS_SYRK((double)(__k), (double)(__n)) +       FADDS_SYRK((double)(__k), (double)(__n)) )
#define FLOPS_SSYRK(__k, __n) (     FMULS_SYRK((double)(__k), (double)(__n)) +       FADDS_SYRK((double)(__k), (double)(__n)) )
 
#define FLOPS_ZHER2K(__k, __n) (6. * FMULS_HER2K((double)(__k), (double)(__n)) + 2.0 * FADDS_HER2K((double)(__k), (double)(__n)) )
#define FLOPS_CHER2K(__k, __n) (6. * FMULS_HER2K((double)(__k), (double)(__n)) + 2.0 * FADDS_HER2K((double)(__k), (double)(__n)) )

#define FLOPS_ZSYR2K(__k, __n) (6. * FMULS_SYR2K((double)(__k), (double)(__n)) + 2.0 * FADDS_SYR2K((double)(__k), (double)(__n)) )
#define FLOPS_CSYR2K(__k, __n) (6. * FMULS_SYR2K((double)(__k), (double)(__n)) + 2.0 * FADDS_SYR2K((double)(__k), (double)(__n)) )
#define FLOPS_DSYR2K(__k, __n) (     FMULS_SYR2K((double)(__k), (double)(__n)) +       FADDS_SYR2K((double)(__k), (double)(__n)) )
#define FLOPS_SSYR2K(__k, __n) (     FMULS_SYR2K((double)(__k), (double)(__n)) +       FADDS_SYR2K((double)(__k), (double)(__n)) )
 
#define FLOPS_ZTRMM(__side, __m, __n) (6. * FMULS_TRMM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_TRMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_CTRMM(__side, __m, __n) (6. * FMULS_TRMM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_TRMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_DTRMM(__side, __m, __n) (     FMULS_TRMM(__side, (double)(__m), (double)(__n)) +       FADDS_TRMM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_STRMM(__side, __m, __n) (     FMULS_TRMM(__side, (double)(__m), (double)(__n)) +       FADDS_TRMM(__side, (double)(__m), (double)(__n)) )
 
#define FLOPS_ZTRSM(__side, __m, __n) (6. * FMULS_TRSM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_TRSM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_CTRSM(__side, __m, __n) (6. * FMULS_TRSM(__side, (double)(__m), (double)(__n)) + 2.0 * FADDS_TRSM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_DTRSM(__side, __m, __n) (     FMULS_TRSM(__side, (double)(__m), (double)(__n)) +       FADDS_TRSM(__side, (double)(__m), (double)(__n)) )
#define FLOPS_STRSM(__side, __m, __n) (     FMULS_TRSM(__side, (double)(__m), (double)(__n)) +       FADDS_TRSM(__side, (double)(__m), (double)(__n)) )
 
/*
 * Lapack
 */
#define FLOPS_ZGETRF(__m, __n) (6. * FMULS_GETRF((double)(__m), (double)(__n)) + 2.0 * FADDS_GETRF((double)(__m), (double)(__n)) )
#define FLOPS_CGETRF(__m, __n) (6. * FMULS_GETRF((double)(__m), (double)(__n)) + 2.0 * FADDS_GETRF((double)(__m), (double)(__n)) )
#define FLOPS_DGETRF(__m, __n) (     FMULS_GETRF((double)(__m), (double)(__n)) +       FADDS_GETRF((double)(__m), (double)(__n)) )
#define FLOPS_SGETRF(__m, __n) (     FMULS_GETRF((double)(__m), (double)(__n)) +       FADDS_GETRF((double)(__m), (double)(__n)) )

#define FLOPS_ZGETRI(__n) (6. * FMULS_GETRI((double)(__n)) + 2.0 * FADDS_GETRI((double)(__n)) )
#define FLOPS_CGETRI(__n) (6. * FMULS_GETRI((double)(__n)) + 2.0 * FADDS_GETRI((double)(__n)) )
#define FLOPS_DGETRI(__n) (     FMULS_GETRI((double)(__n)) +       FADDS_GETRI((double)(__n)) )
#define FLOPS_SGETRI(__n) (     FMULS_GETRI((double)(__n)) +       FADDS_GETRI((double)(__n)) )

#define FLOPS_ZGETRS(__n, __nrhs) (6. * FMULS_GETRS((double)(__n), (double)(__nrhs)) + 2.0 * FADDS_GETRS((double)(__n), (double)(__nrhs)) )
#define FLOPS_CGETRS(__n, __nrhs) (6. * FMULS_GETRS((double)(__n), (double)(__nrhs)) + 2.0 * FADDS_GETRS((double)(__n), (double)(__nrhs)) )
#define FLOPS_DGETRS(__n, __nrhs) (     FMULS_GETRS((double)(__n), (double)(__nrhs)) +       FADDS_GETRS((double)(__n), (double)(__nrhs)) )
#define FLOPS_SGETRS(__n, __nrhs) (     FMULS_GETRS((double)(__n), (double)(__nrhs)) +       FADDS_GETRS((double)(__n), (double)(__nrhs)) )

#define FLOPS_ZPOTRF(__n) (6. * FMULS_POTRF((double)(__n)) + 2.0 * FADDS_POTRF((double)(__n)) )
#define FLOPS_CPOTRF(__n) (6. * FMULS_POTRF((double)(__n)) + 2.0 * FADDS_POTRF((double)(__n)) )
#define FLOPS_DPOTRF(__n) (     FMULS_POTRF((double)(__n)) +       FADDS_POTRF((double)(__n)) )
#define FLOPS_SPOTRF(__n) (     FMULS_POTRF((double)(__n)) +       FADDS_POTRF((double)(__n)) )

#define FLOPS_ZPOTRI(__n) (6. * FMULS_POTRI((double)(__n)) + 2.0 * FADDS_POTRI((double)(__n)) )
#define FLOPS_CPOTRI(__n) (6. * FMULS_POTRI((double)(__n)) + 2.0 * FADDS_POTRI((double)(__n)) )
#define FLOPS_DPOTRI(__n) (     FMULS_POTRI((double)(__n)) +       FADDS_POTRI((double)(__n)) )
#define FLOPS_SPOTRI(__n) (     FMULS_POTRI((double)(__n)) +       FADDS_POTRI((double)(__n)) )

#define FLOPS_ZPOTRS(__n, __nrhs) (6. * FMULS_POTRS((double)(__n), (double)(__nrhs)) + 2.0 * FADDS_POTRS((double)(__n), (double)(__nrhs)) )
#define FLOPS_CPOTRS(__n, __nrhs) (6. * FMULS_POTRS((double)(__n), (double)(__nrhs)) + 2.0 * FADDS_POTRS((double)(__n), (double)(__nrhs)) )
#define FLOPS_DPOTRS(__n, __nrhs) (     FMULS_POTRS((double)(__n), (double)(__nrhs)) +       FADDS_POTRS((double)(__n), (double)(__nrhs)) )
#define FLOPS_SPOTRS(__n, __nrhs) (     FMULS_POTRS((double)(__n), (double)(__nrhs)) +       FADDS_POTRS((double)(__n), (double)(__nrhs)) )

#define FLOPS_ZGEQRF(__m, __n) (6. * FMULS_GEQRF((double)(__m), (double)(__n)) + 2.0 * FADDS_GEQRF((double)(__m), (double)(__n)) )
#define FLOPS_CGEQRF(__m, __n) (6. * FMULS_GEQRF((double)(__m), (double)(__n)) + 2.0 * FADDS_GEQRF((double)(__m), (double)(__n)) )
#define FLOPS_DGEQRF(__m, __n) (     FMULS_GEQRF((double)(__m), (double)(__n)) +       FADDS_GEQRF((double)(__m), (double)(__n)) )
#define FLOPS_SGEQRF(__m, __n) (     FMULS_GEQRF((double)(__m), (double)(__n)) +       FADDS_GEQRF((double)(__m), (double)(__n)) )

#define FLOPS_ZGEQLF(__m, __n) (6. * FMULS_GEQLF((double)(__m), (double)(__n)) + 2.0 * FADDS_GEQLF((double)(__m), (double)(__n)) )
#define FLOPS_CGEQLF(__m, __n) (6. * FMULS_GEQLF((double)(__m), (double)(__n)) + 2.0 * FADDS_GEQLF((double)(__m), (double)(__n)) )
#define FLOPS_DGEQLF(__m, __n) (     FMULS_GEQLF((double)(__m), (double)(__n)) +       FADDS_GEQLF((double)(__m), (double)(__n)) )
#define FLOPS_SGEQLF(__m, __n) (     FMULS_GEQLF((double)(__m), (double)(__n)) +       FADDS_GEQLF((double)(__m), (double)(__n)) )

#define FLOPS_ZGERQF(__m, __n) (6. * FMULS_GERQF((double)(__m), (double)(__n)) + 2.0 * FADDS_GERQF((double)(__m), (double)(__n)) )
#define FLOPS_CGERQF(__m, __n) (6. * FMULS_GERQF((double)(__m), (double)(__n)) + 2.0 * FADDS_GERQF((double)(__m), (double)(__n)) )
#define FLOPS_DGERQF(__m, __n) (     FMULS_GERQF((double)(__m), (double)(__n)) +       FADDS_GERQF((double)(__m), (double)(__n)) )
#define FLOPS_SGERQF(__m, __n) (     FMULS_GERQF((double)(__m), (double)(__n)) +       FADDS_GERQF((double)(__m), (double)(__n)) )

#define FLOPS_ZGELQF(__m, __n) (6. * FMULS_GELQF((double)(__m), (double)(__n)) + 2.0 * FADDS_GELQF((double)(__m), (double)(__n)) )
#define FLOPS_CGELQF(__m, __n) (6. * FMULS_GELQF((double)(__m), (double)(__n)) + 2.0 * FADDS_GELQF((double)(__m), (double)(__n)) )
#define FLOPS_DGELQF(__m, __n) (     FMULS_GELQF((double)(__m), (double)(__n)) +       FADDS_GELQF((double)(__m), (double)(__n)) )
#define FLOPS_SGELQF(__m, __n) (     FMULS_GELQF((double)(__m), (double)(__n)) +       FADDS_GELQF((double)(__m), (double)(__n)) )

#define FLOPS_ZUNGQR(__m, __n, __k) (6. * FMULS_UNGQR((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGQR((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CUNGQR(__m, __n, __k) (6. * FMULS_UNGQR((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGQR((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DUNGQR(__m, __n, __k) (     FMULS_UNGQR((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGQR((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SUNGQR(__m, __n, __k) (     FMULS_UNGQR((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGQR((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZUNGQL(__m, __n, __k) (6. * FMULS_UNGQL((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGQL((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CUNGQL(__m, __n, __k) (6. * FMULS_UNGQL((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGQL((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DUNGQL(__m, __n, __k) (     FMULS_UNGQL((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGQL((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SUNGQL(__m, __n, __k) (     FMULS_UNGQL((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGQL((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZORGQR(__m, __n, __k) (6. * FMULS_ORGQR((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGQR((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CORGQR(__m, __n, __k) (6. * FMULS_ORGQR((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGQR((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DORGQR(__m, __n, __k) (     FMULS_ORGQR((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGQR((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SORGQR(__m, __n, __k) (     FMULS_ORGQR((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGQR((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZORGQL(__m, __n, __k) (6. * FMULS_ORGQL((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGQL((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CORGQL(__m, __n, __k) (6. * FMULS_ORGQL((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGQL((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DORGQL(__m, __n, __k) (     FMULS_ORGQL((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGQL((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SORGQL(__m, __n, __k) (     FMULS_ORGQL((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGQL((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZUNGRQ(__m, __n, __k) (6. * FMULS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CUNGRQ(__m, __n, __k) (6. * FMULS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DUNGRQ(__m, __n, __k) (     FMULS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SUNGRQ(__m, __n, __k) (     FMULS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGRQ((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZUNGLQ(__m, __n, __k) (6. * FMULS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CUNGLQ(__m, __n, __k) (6. * FMULS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DUNGLQ(__m, __n, __k) (     FMULS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SUNGLQ(__m, __n, __k) (     FMULS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_UNGLQ((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZORGRQ(__m, __n, __k) (6. * FMULS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CORGRQ(__m, __n, __k) (6. * FMULS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DORGRQ(__m, __n, __k) (     FMULS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SORGRQ(__m, __n, __k) (     FMULS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGRQ((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZORGLQ(__m, __n, __k) (6. * FMULS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_CORGLQ(__m, __n, __k) (6. * FMULS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) + 2.0 * FADDS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_DORGLQ(__m, __n, __k) (     FMULS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SORGLQ(__m, __n, __k) (     FMULS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) +       FADDS_ORGLQ((double)(__m), (double)(__n), (double)(__k)) )

#define FLOPS_ZGEQRS(__m, __n, __nrhs) (6. * FMULS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) + 2.0 * FADDS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) )
#define FLOPS_CGEQRS(__m, __n, __nrhs) (6. * FMULS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) + 2.0 * FADDS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) )
#define FLOPS_DGEQRS(__m, __n, __nrhs) (     FMULS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) +       FADDS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) )
#define FLOPS_SGEQRS(__m, __n, __nrhs) (     FMULS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) +       FADDS_GEQRS((double)(__m), (double)(__n), (double)(__nrhs)) )

#define FLOPS_ZTRTRI(__n) (6. * FMULS_TRTRI((double)(__n)) + 2.0 * FADDS_TRTRI((double)(__n)) )
#define FLOPS_CTRTRI(__n) (6. * FMULS_TRTRI((double)(__n)) + 2.0 * FADDS_TRTRI((double)(__n)) )
#define FLOPS_DTRTRI(__n) (     FMULS_TRTRI((double)(__n)) +       FADDS_TRTRI((double)(__n)) )
#define FLOPS_STRTRI(__n) (     FMULS_TRTRI((double)(__n)) +       FADDS_TRTRI((double)(__n)) )

#define FLOPS_ZGEHRD(__n) (6. * FMULS_GEHRD((double)(__n)) + 2.0 * FADDS_GEHRD((double)(__n)) )
#define FLOPS_CGEHRD(__n) (6. * FMULS_GEHRD((double)(__n)) + 2.0 * FADDS_GEHRD((double)(__n)) )
#define FLOPS_DGEHRD(__n) (     FMULS_GEHRD((double)(__n)) +       FADDS_GEHRD((double)(__n)) )
#define FLOPS_SGEHRD(__n) (     FMULS_GEHRD((double)(__n)) +       FADDS_GEHRD((double)(__n)) )

#define FLOPS_ZHETRD(__n) (6. * FMULS_HETRD((double)(__n)) + 2.0 * FADDS_HETRD((double)(__n)) )
#define FLOPS_CHETRD(__n) (6. * FMULS_HETRD((double)(__n)) + 2.0 * FADDS_HETRD((double)(__n)) )

#define FLOPS_ZSYTRD(__n) (6. * FMULS_SYTRD((double)(__n)) + 2.0 * FADDS_SYTRD((double)(__n)) )
#define FLOPS_CSYTRD(__n) (6. * FMULS_SYTRD((double)(__n)) + 2.0 * FADDS_SYTRD((double)(__n)) )
#define FLOPS_DSYTRD(__n) (     FMULS_SYTRD((double)(__n)) +       FADDS_SYTRD((double)(__n)) )
#define FLOPS_SSYTRD(__n) (     FMULS_SYTRD((double)(__n)) +       FADDS_SYTRD((double)(__n)) )

#define FLOPS_ZGEBRD(__m, __n) (6. * FMULS_GEBRD((double)(__m), (double)(__n)) + 2.0 * FADDS_GEBRD((double)(__m), (double)(__n)) )
#define FLOPS_CGEBRD(__m, __n) (6. * FMULS_GEBRD((double)(__m), (double)(__n)) + 2.0 * FADDS_GEBRD((double)(__m), (double)(__n)) )
#define FLOPS_DGEBRD(__m, __n) (     FMULS_GEBRD((double)(__m), (double)(__n)) +       FADDS_GEBRD((double)(__m), (double)(__n)) )
#define FLOPS_SGEBRD(__m, __n) (     FMULS_GEBRD((double)(__m), (double)(__n)) +       FADDS_GEBRD((double)(__m), (double)(__n)) )

#endif /* _FLOPS_H_ */
