/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#if defined(__GNUC__)
#include <stdint.h>
#endif /* __GNUC__ */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

/* 
 * typedef comming from fortran.h file provided in $CUDADIR/src directory
 * it will probably change with future release of cublas when they will use 64bits address
 */
typedef size_t devptr_t;

#define PRECISION_z

#ifdef PGI_FORTRAN
#define DEVPTR(__ptr) ((cuDoubleComplex*)(__ptr))
#else
#define DEVPTR(__ptr) ((cuDoubleComplex*)(uintptr_t)(*(__ptr)))
#endif


#ifndef MAGMA_FORTRAN_NAME
#if defined(ADD_)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_
#elif defined(NOCHANGE)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname
#elif defined(UPCASE)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  MAGMAF_##UCNAME
#endif
#endif

#ifndef MAGMA_GPU_FORTRAN_NAME
#if defined(ADD_)
#define MAGMA_GPU_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_gpu_
#elif defined(NOCHANGE)
#define MAGMA_GPU_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_gpu
#elif defined(UPCASE)
#define MAGMA_GPU_FORTRAN_NAME(lcname, UCNAME)  MAGMAF_##UCNAME##_GPU
#endif
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
#define MAGMAF_ZGEBRD  MAGMA_FORTRAN_NAME(zgebrd,  ZGEBRD ) 
#define MAGMAF_ZGEHRD2 MAGMA_FORTRAN_NAME(zgehrd2, ZGEHRD2)
#define MAGMAF_ZGEHRD  MAGMA_FORTRAN_NAME(zgehrd,  ZGEHRD )
#define MAGMAF_ZGELQF  MAGMA_FORTRAN_NAME(zgelqf,  ZGELQF )
#define MAGMAF_ZGEQLF  MAGMA_FORTRAN_NAME(zgeqlf,  ZGEQLF )
#define MAGMAF_ZGEQRF  MAGMA_FORTRAN_NAME(zgeqrf,  ZGEQRF )
#define MAGMAF_ZGESV   MAGMA_FORTRAN_NAME(zgesv,   ZGESV  )
#define MAGMAF_ZGETRF  MAGMA_FORTRAN_NAME(zgetrf,  ZGETRF )
#define MAGMAF_ZLATRD  MAGMA_FORTRAN_NAME(zlatrd,  ZLATRD )
#define MAGMAF_ZLAHR2  MAGMA_FORTRAN_NAME(zlahr2,  ZLAHR2 )
#define MAGMAF_ZLAHRU  MAGMA_FORTRAN_NAME(zlahru,  ZLAHRU )
#define MAGMAF_ZPOSV   MAGMA_FORTRAN_NAME(zposv,   ZPOSV  )
#define MAGMAF_ZPOTRF  MAGMA_FORTRAN_NAME(zpotrf,  ZPOTRF )
#define MAGMAF_ZHETRD  MAGMA_FORTRAN_NAME(zhetrd,  ZHETRD )
#define MAGMAF_ZUNGQR  MAGMA_FORTRAN_NAME(zungqr,  ZUNGQR )
#define MAGMAF_ZUNMQR  MAGMA_FORTRAN_NAME(zunmqr,  ZUNMQR )
#define MAGMAF_ZUNMTR  MAGMA_FORTRAN_NAME(zunmtr,  ZUNMTR )
#define MAGMAF_ZUNGHR  MAGMA_FORTRAN_NAME(zunghr,  ZUNGHR )
#define MAGMAF_ZGEEV   MAGMA_FORTRAN_NAME(zgeev,   ZGEEV  )
#define MAGMAF_ZGESVD  MAGMA_FORTRAN_NAME(zgesvd,  ZGESVD )
#define MAGMAF_ZHEEVD  MAGMA_FORTRAN_NAME(zheevd,  ZHEEVD )
#define MAGMAF_ZHEGVD  MAGMA_FORTRAN_NAME(zhegvd,  ZHEGVD )

/* //////////////////////////////////////////////////////////////////////////// 
 -- MAGMA function definitions / Data on GPU
*/
#define MAGMAF_ZGELS_GPU   MAGMA_GPU_FORTRAN_NAME(zgels,   ZGELS  )
#define MAGMAF_ZGEQRF_GPU  MAGMA_GPU_FORTRAN_NAME(zgeqrf,  ZGEQRF ) 
#define MAGMAF_ZGEQRF2_GPU MAGMA_GPU_FORTRAN_NAME(zgeqrf2, ZGEQRF2)
#define MAGMAF_ZGEQRF3_GPU MAGMA_GPU_FORTRAN_NAME(zgeqrf3, ZGEQRF3)
#define MAGMAF_ZGEQRS_GPU  MAGMA_GPU_FORTRAN_NAME(zgeqrs,  ZGEQRS ) 
#define MAGMAF_ZGEQRS3_GPU MAGMA_GPU_FORTRAN_NAME(zgeqrs3, ZGEQRS3) 
#define MAGMAF_ZGESSM_GPU  MAGMA_GPU_FORTRAN_NAME(zgessm,  ZGESSM ) 
#define MAGMAF_ZGESV_GPU   MAGMA_GPU_FORTRAN_NAME(zgesv,   ZGESV  )  
#define MAGMAF_ZGETRF_INCPIV_GPU  MAGMA_GPU_FORTRAN_NAME(zgetrf_incpiv,  ZGETRF_INCPIV ) 
#define MAGMAF_ZGETRF_GPU  MAGMA_GPU_FORTRAN_NAME(zgetrf,  ZGETRF ) 
#define MAGMAF_ZGETRS_GPU  MAGMA_GPU_FORTRAN_NAME(zgetrs,  ZGETRS ) 
#define MAGMAF_ZLABRD_GPU  MAGMA_GPU_FORTRAN_NAME(zlabrd,  ZLABRD ) 
#define MAGMAF_ZLARFB_GPU  MAGMA_GPU_FORTRAN_NAME(zlarfb,  ZLARFB ) 
#define MAGMAF_ZPOSV_GPU   MAGMA_GPU_FORTRAN_NAME(zposv,   ZPOSV  )  
#define MAGMAF_ZPOTRF_GPU  MAGMA_GPU_FORTRAN_NAME(zpotrf,  ZPOTRF ) 
#define MAGMAF_ZPOTRS_GPU  MAGMA_GPU_FORTRAN_NAME(zpotrs,  ZPOTRS ) 
#define MAGMAF_ZSSSSM_GPU  MAGMA_GPU_FORTRAN_NAME(zssssm,  ZSSSSM ) 
#define MAGMAF_ZTSTRF_GPU  MAGMA_GPU_FORTRAN_NAME(ztstrf,  ZTSTRF ) 
#define MAGMAF_ZUNGQR_GPU  MAGMA_GPU_FORTRAN_NAME(zungqr,  ZUNGQR ) 
#define MAGMAF_ZUNMQR_GPU  MAGMA_GPU_FORTRAN_NAME(zunmqr,  ZUNMQR ) 

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *  FORTRAN API - math functions (simple interface)
 **/

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
void MAGMAF_ZGEBRD( magma_int_t *m, magma_int_t *n, cuDoubleComplex *A, 
                    magma_int_t *lda, double *d, double *e,
                    cuDoubleComplex *tauq, cuDoubleComplex *taup, 
                    cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info)
{
    magma_zgebrd( *m, *n, A, 
                  *lda, d, e,
                  tauq, taup, 
                  work, *lwork, info);
}
    
void MAGMAF_ZGEHRD2(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi,
                    cuDoubleComplex *A, magma_int_t *lda, cuDoubleComplex *tau, 
                    cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info)
{
    magma_zgehrd2(*n, *ilo, *ihi,
                  A, *lda, tau, 
                  work, lwork, info);
}
    
void MAGMAF_ZGEHRD( magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi,
            cuDoubleComplex *A, magma_int_t *lda, cuDoubleComplex *tau,
            cuDoubleComplex *work, magma_int_t *lwork,
            cuDoubleComplex *d_T, magma_int_t *info)
{
  magma_zgehrd( *n, *ilo, *ihi,
        A, *lda, tau,
        work, *lwork,
        d_T, info);
}

void MAGMAF_ZGELQF( magma_int_t *m, magma_int_t *n, 
                    cuDoubleComplex *A,    magma_int_t *lda,   cuDoubleComplex *tau, 
                    cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info)
{
    magma_zgelqf( *m, *n, 
                  A,    *lda,   tau, 
                  work, *lwork, info);
}

void MAGMAF_ZGEQLF( magma_int_t *m, magma_int_t *n, 
                    cuDoubleComplex *A,    magma_int_t *lda,   cuDoubleComplex *tau, 
                    cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info)
{
    magma_zgeqlf( *m, *n, 
                  A,    *lda,   tau, 
                  work, *lwork, info);
}

void MAGMAF_ZGEQRF( magma_int_t *m, magma_int_t *n, cuDoubleComplex *A, 
                    magma_int_t *lda, cuDoubleComplex *tau, cuDoubleComplex *work, 
                    magma_int_t *lwork, magma_int_t *info)
{
    magma_zgeqrf( *m, *n, A, 
                  *lda, tau, work, 
                  *lwork, info);
}

void MAGMAF_ZGESV ( magma_int_t *n, magma_int_t *nrhs,
                    cuDoubleComplex *A, magma_int_t *lda, magma_int_t *ipiv,
                    cuDoubleComplex *B, magma_int_t *ldb, magma_int_t *info)
{
    magma_zgesv(  *n, *nrhs,
                  A, *lda, ipiv,
                  B, *ldb,
                  info);
}
    
void MAGMAF_ZGETRF( magma_int_t *m, magma_int_t *n, cuDoubleComplex *A, 
                    magma_int_t *lda, magma_int_t *ipiv, 
                    magma_int_t *info)
{
    magma_zgetrf( *m, *n, A, 
                  *lda, ipiv, 
                  info);
}

// void MAGMAF_ZLATRD( char *uplo, magma_int_t *n, magma_int_t *nb, cuDoubleComplex *a, 
//                     magma_int_t *lda, double *e, cuDoubleComplex *tau, 
//                     cuDoubleComplex *w, magma_int_t *ldw,
//                     cuDoubleComplex *da, magma_int_t *ldda, 
//                     cuDoubleComplex *dw, magma_int_t *lddw)
// {
//     magma_zlatrd( uplo[0], *n, *nb, a, 
//                   *lda, e, tau, 
//                   w, *ldw,
//                   da, *ldda, 
//                   dw, *lddw);
// }

  /* This has nothing to do here, it should be a GPU function */
// void MAGMAF_ZLAHR2( magma_int_t *m, magma_int_t *n, magma_int_t *nb, 
//                     cuDoubleComplex *da, cuDoubleComplex *dv, cuDoubleComplex *a, 
//                     magma_int_t *lda, cuDoubleComplex *tau, cuDoubleComplex *t, 
//                     magma_int_t *ldt, cuDoubleComplex *y, magma_int_t *ldy)
// {
//     magma_zlahr2( *m, *n, *nb, 
//                   da, dv, a, 
//                   *lda, tau, t, 
//                   *ldt, y, *ldy);
// }

// void MAGMAF_ZLAHRU( magma_int_t *m, magma_int_t *n, magma_int_t *nb, 
//                     cuDoubleComplex *a, magma_int_t *lda, 
//                     cuDoubleComplex *da, cuDoubleComplex *y, 
//                     cuDoubleComplex *v, cuDoubleComplex *t, 
//                     cuDoubleComplex *dwork)
// {
//     magma_zlahru( *m, *n, *nb, 
//                   a, *lda, 
//                   da, y, 
//                   v, t, 
//                   dwork);
// }

void MAGMAF_ZPOSV(  char *uplo, magma_int_t *n, magma_int_t *nrhs,
                    cuDoubleComplex *A, magma_int_t *lda,
                    cuDoubleComplex *B, magma_int_t *ldb, magma_int_t *info)
{
    magma_zposv(  uplo[0], *n, *nrhs,
                  A, *lda,
                  B, *ldb, info);
}

void MAGMAF_ZPOTRF( char *uplo, magma_int_t *n, cuDoubleComplex *A, 
                    magma_int_t *lda, magma_int_t *info)
{
    magma_zpotrf( uplo[0], *n, A, 
                  *lda, info);
}

void MAGMAF_ZHETRD( char *uplo, magma_int_t *n, cuDoubleComplex *A, 
                    magma_int_t *lda, double *d, double *e, 
                    cuDoubleComplex *tau, cuDoubleComplex *work, magma_int_t *lwork, 
                    magma_int_t *info)
{
    magma_zhetrd( uplo[0], *n, A, 
                  *lda, d, e, 
                  tau, work, *lwork, 
                  info);
}

// void MAGMAF_ZUNGQR( magma_int_t *m, magma_int_t *n, magma_int_t *k,
//                     cuDoubleComplex *a, magma_int_t *lda,
//                     cuDoubleComplex *tau, cuDoubleComplex *dwork,
//                     magma_int_t *nb, magma_int_t *info )
// {
//     magma_zungqr( *m, *n, *k,
//                   a, *lda,
//                   tau, dwork,
//                   *nb, info );
// }

void MAGMAF_ZUNMQR( char *side, char *trans, 
                    magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                    cuDoubleComplex *a, magma_int_t *lda, cuDoubleComplex *tau, 
                    cuDoubleComplex *c, magma_int_t *ldc, 
                    cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info)
{
    magma_zunmqr( side[0], trans[0], 
                  *m, *n, *k, 
                  a, *lda, tau, 
                  c, *ldc, 
                  work, *lwork, info);
}

void MAGMAF_ZUNMTR( char *side, char *uplo, char *trans,
                    magma_int_t *m, magma_int_t *n,
                    cuDoubleComplex *a,    magma_int_t *lda,
                    cuDoubleComplex *tau,
                    cuDoubleComplex *c,    magma_int_t *ldc,
                    cuDoubleComplex *work, magma_int_t *lwork,
                    magma_int_t *info)
{
    magma_zunmtr( side[0], uplo[0], trans[0],
                  *m, *n,
                  a,    *lda,
                  tau,
                  c,    *ldc,
                  work, *lwork,
                  info);
}

// void MAGMAF_ZUNGHR( magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi,
//                     cuDoubleComplex *a, magma_int_t *lda,
//                     cuDoubleComplex *tau,
//                     cuDoubleComplex *dT, magma_int_t *nb,
//                     magma_int_t *info)
// {
//     magma_zunghr( *n, *ilo, *ihi,
//                   a, *lda,
//                   tau,
//                   dT, *nb,
//                   info);
// }

#if defined(PRECISION_z) || defined(PRECISION_c)
void MAGMAF_ZGEEV( char *jobvl, char *jobvr, magma_int_t *n,
                   cuDoubleComplex *a, magma_int_t *lda,
                   cuDoubleComplex *w,
                   cuDoubleComplex *vl, magma_int_t *ldvl,
                   cuDoubleComplex *vr, magma_int_t *ldvr,
                   cuDoubleComplex *work, magma_int_t *lwork,
                   double *rwork, magma_int_t *info)
{
    magma_zgeev( jobvl[0], jobvr[0], *n,
                 a, *lda,
                 w,
                 vl, *ldvl,
                 vr, *ldvr,
                 work, *lwork,
                 rwork, info);
}

void MAGMAF_ZGESVD( char *jobu, char *jobvt, magma_int_t *m, magma_int_t *n,
                    cuDoubleComplex *a,    magma_int_t *lda, double *s, 
                    cuDoubleComplex *u,    magma_int_t *ldu, 
                    cuDoubleComplex *vt,   magma_int_t *ldvt,
                    cuDoubleComplex *work, magma_int_t *lwork,
                    double *rwork, magma_int_t *info )
{
    magma_zgesvd( jobu[0], jobvt[0], *m, *n,
                  a,    *lda, s, 
                  u,    *ldu, 
                  vt,   *ldvt,
                  work, *lwork,
                  rwork, info );
}
    
void MAGMAF_ZHEEVD( char *jobz, char *uplo, magma_int_t *n,
                    cuDoubleComplex *a,     magma_int_t *lda, double *w,
                    cuDoubleComplex *work,  magma_int_t *lwork,
                    double          *rwork, magma_int_t *lrwork,
                    magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info)
{
    magma_zheevd( jobz[0], uplo[0], *n,
                  a, *lda, w,
                  work, *lwork,
                  rwork, *lrwork,
                  iwork, *liwork, info);
}

void MAGMAF_ZHEGVD(magma_int_t *itype, char *jobz, char *uplo, magma_int_t *n,
           cuDoubleComplex *a, magma_int_t *lda, 
           cuDoubleComplex *b, magma_int_t *ldb,
           double *w, cuDoubleComplex *work, magma_int_t *lwork,
           double *rwork, magma_int_t *lrwork,
           magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info)
{
  magma_zhegvd( *itype, jobz[0], uplo[0], *n,
        a, *lda, b, *ldb,
        w, work, *lwork,
        rwork, *lrwork,
        iwork, *liwork, info);
}
    
#else
void MAGMAF_ZGEEV( char *jobvl, char *jobvr, magma_int_t *n,
                   cuDoubleComplex *a,    magma_int_t *lda,
                   cuDoubleComplex *wr, cuDoubleComplex *wi,
                   cuDoubleComplex *vl,   magma_int_t *ldvl,
                   cuDoubleComplex *vr,   magma_int_t *ldvr,
                   cuDoubleComplex *work, magma_int_t *lwork,
                   magma_int_t *info)
{
    magma_zgeev( jobvl[0], jobvr[0], *n,
                 a,    *lda,
                 wr, wi,
                 vl,   *ldvl,
                 vr,   *ldvr,
                 work, *lwork,
                 info);
}

void MAGMAF_ZGESVD( char *jobu, char *jobvt, magma_int_t *m, magma_int_t *n,
                    cuDoubleComplex *a,    magma_int_t *lda, double *s,
                    cuDoubleComplex *u,    magma_int_t *ldu, 
                    cuDoubleComplex *vt,   magma_int_t *ldvt,
                    cuDoubleComplex *work, magma_int_t *lwork,
                    magma_int_t *info )
{
    magma_zgesvd( jobu[0], jobvt[0], *m, *n,
                  a,    *lda, s, 
                  u,    *ldu, 
                  vt,   *ldvt,
                  work, *lwork,
                  info );
}

void MAGMAF_ZHEEVD( char *jobz, char *uplo, magma_int_t *n,
                    cuDoubleComplex *a, magma_int_t *lda, double *w,
                    cuDoubleComplex *work, magma_int_t *lwork,
                    magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info)
{
    magma_zheevd( jobz[0], uplo[0], *n,
                  a, *lda, w,
                  work, *lwork,
                  iwork, *liwork, info);
}

void MAGMAF_ZHEGVD(magma_int_t *itype, char *jobz, char *uplo, magma_int_t *n,
                   cuDoubleComplex *a, magma_int_t *lda,
                   cuDoubleComplex *b, magma_int_t *ldb,
                   double *w, cuDoubleComplex *work, magma_int_t *lwork,
                   magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info)
{
  magma_zhegvd( *itype, jobz[0], uplo[0], *n,
                a, *lda, b, *ldb,
        w, work, *lwork,
                iwork, *liwork, info);
}


#endif

/* //////////////////////////////////////////////////////////////////////////// 
 -- MAGMA function definitions / Data on GPU
*/
void MAGMAF_ZGELS_GPU(  char *trans, magma_int_t *m, magma_int_t *n, magma_int_t *nrhs,
                        devptr_t *dA,    magma_int_t *ldda, 
                        devptr_t *dB,    magma_int_t *lddb, 
                        cuDoubleComplex *hwork, magma_int_t *lwork, 
                        magma_int_t *info)
{
    magma_zgels_gpu(  trans[0], *m, *n, *nrhs, 
                      DEVPTR(dA),    *ldda,  
                      DEVPTR(dB),    *lddb,  
                      hwork, *lwork,  info);
}

void MAGMAF_ZGEQRF_GPU( magma_int_t *m, magma_int_t *n, 
                        devptr_t *dA,  magma_int_t *ldda, 
                        cuDoubleComplex *tau, devptr_t *dT, 
                        magma_int_t *info)
{
    magma_zgeqrf_gpu( *m, *n,  
                      DEVPTR(dA),  *ldda,  
                      tau, 
                      DEVPTR(dT),  info);
}

void MAGMAF_ZGEQRF2_GPU(magma_int_t *m, magma_int_t *n, 
                        devptr_t *dA,  magma_int_t *ldda, 
                        cuDoubleComplex *tau, magma_int_t *info)
{
    magma_zgeqrf2_gpu(*m, *n,  
                      DEVPTR(dA),  *ldda,  
                      tau, info); 
}

void MAGMAF_ZGEQRF3_GPU(magma_int_t *m, magma_int_t *n, 
                        devptr_t *dA,  magma_int_t *ldda, 
                        cuDoubleComplex *tau, devptr_t *dT,
                        magma_int_t *info)
{
    magma_zgeqrf3_gpu(*m, *n,  
                      DEVPTR(dA),  *ldda,  
                      tau, DEVPTR(dT), info); 
}

void MAGMAF_ZGEQRS_GPU( magma_int_t *m, magma_int_t *n, magma_int_t *nrhs, 
                        devptr_t *dA,     magma_int_t *ldda, 
                        cuDoubleComplex *tau,   devptr_t *dT,
                        devptr_t *dB,    magma_int_t *lddb,
                        cuDoubleComplex *hwork, magma_int_t *lhwork, 
                        magma_int_t *info)
{
    magma_zgeqrs_gpu( *m, *n, *nrhs,  
                      DEVPTR(dA),     *ldda,  
                      tau,
                      DEVPTR(dT), 
                      DEVPTR(dB),    *lddb, 
                      hwork, *lhwork,  info);
}

void MAGMAF_ZGEQRS3_GPU(magma_int_t *m, magma_int_t *n, magma_int_t *nrhs, 
                        devptr_t *dA,     magma_int_t *ldda, 
                        cuDoubleComplex *tau,   devptr_t *dT,
                        devptr_t *dB,    magma_int_t *lddb,
                        cuDoubleComplex *hwork, magma_int_t *lhwork, 
                        magma_int_t *info)
{
    magma_zgeqrs3_gpu(*m, *n, *nrhs,  
                      DEVPTR(dA),     *ldda,  
                      tau,
                      DEVPTR(dT), 
                      DEVPTR(dB),    *lddb, 
                      hwork, *lhwork,  info);
}

void MAGMAF_ZGESSM_GPU( char *storev, magma_int_t *m, magma_int_t *n, magma_int_t *k, magma_int_t *ib, 
                        magma_int_t *ipiv, 
                        devptr_t *dL1, magma_int_t *lddl1, 
                        devptr_t *dL,  magma_int_t *lddl, 
                        devptr_t *dA,  magma_int_t *ldda, 
                        magma_int_t *info)
{
    magma_zgessm_gpu( storev[0], *m, *n, *k, *ib, ipiv,  
                      DEVPTR(dL1), *lddl1,  
                      DEVPTR(dL),  *lddl,  
                      DEVPTR(dA),  *ldda,  info);
}

void MAGMAF_ZGESV_GPU(  magma_int_t *n, magma_int_t *nrhs, 
                        devptr_t *dA, magma_int_t *ldda, magma_int_t *ipiv, 
                        devptr_t *dB, magma_int_t *lddb, magma_int_t *info)
{
    magma_zgesv_gpu(  *n, *nrhs,  
                      DEVPTR(dA), *ldda, ipiv,  
                      DEVPTR(dB), *lddb, info);
}

void MAGMAF_ZGETRF_GPU( magma_int_t *m, magma_int_t *n, 
                        devptr_t *dA, magma_int_t *ldda, 
                        magma_int_t *ipiv, magma_int_t *info)
{
    magma_zgetrf_gpu( *m, *n,  
                      DEVPTR(dA), *ldda, ipiv, info);
}

void MAGMAF_ZGETRS_GPU( char *trans, magma_int_t *n, magma_int_t *nrhs, 
                        devptr_t *dA, magma_int_t *ldda, magma_int_t *ipiv, 
                        devptr_t *dB, magma_int_t *lddb, magma_int_t *info)
{
    magma_zgetrs_gpu( trans[0], *n, *nrhs,  
                      DEVPTR(dA), *ldda, ipiv,  
                      DEVPTR(dB), *lddb, info);
}

void MAGMAF_ZLABRD_GPU( magma_int_t *m, magma_int_t *n, magma_int_t *nb, 
                        cuDoubleComplex *a, magma_int_t *lda, devptr_t *da, magma_int_t *ldda,
                        double *d, double *e, cuDoubleComplex *tauq, cuDoubleComplex *taup,  
                        cuDoubleComplex *x, magma_int_t *ldx, devptr_t *dx, magma_int_t *lddx, 
                        cuDoubleComplex *y, magma_int_t *ldy, devptr_t *dy, magma_int_t *lddy)
{
    magma_zlabrd_gpu( *m, *n, *nb,  
                      a, *lda, DEVPTR(da), *ldda, 
                      d, e, tauq, taup,   
                      x, *ldx, DEVPTR(dx), *lddx,  
                      y, *ldy, DEVPTR(dy), *lddy);
}

void MAGMAF_ZLARFB_GPU( char *side, char *trans, char *direct, char *storev, 
                        magma_int_t *m, magma_int_t *n, magma_int_t *k,
                        devptr_t *dv, magma_int_t *ldv, devptr_t *dt,    magma_int_t *ldt, 
                        devptr_t *dc, magma_int_t *ldc, devptr_t *dowrk, magma_int_t *ldwork )
{
    magma_zlarfb_gpu( side[0], trans[0], direct[0], storev[0],  *m, *n, *k, 
                      DEVPTR(dv), *ldv, DEVPTR(dt),    *ldt,  
                      DEVPTR(dc), *ldc, DEVPTR(dowrk), *ldwork);
}

void MAGMAF_ZPOSV_GPU(  char *uplo, magma_int_t *n, magma_int_t *nrhs, 
                        devptr_t *dA, magma_int_t *ldda, 
                        devptr_t *dB, magma_int_t *lddb, magma_int_t *info)
{
    magma_zposv_gpu(  uplo[0], *n, *nrhs,  
                      DEVPTR(dA), *ldda,  
                      DEVPTR(dB), *lddb, info);
}

void MAGMAF_ZPOTRF_GPU( char *uplo,  magma_int_t *n, 
                        devptr_t *dA, magma_int_t *ldda, magma_int_t *info)
{
    magma_zpotrf_gpu( uplo[0],  *n,  
                      DEVPTR(dA), *ldda, info); }

void MAGMAF_ZPOTRS_GPU( char *uplo,  magma_int_t *n, magma_int_t *nrhs, 
                        devptr_t *dA, magma_int_t *ldda, 
                        devptr_t *dB, magma_int_t *lddb, magma_int_t *info)
{
    magma_zpotrs_gpu( uplo[0],  *n, *nrhs,  
                      DEVPTR(dA), *ldda,  
                      DEVPTR(dB), *lddb, info);
}

void MAGMAF_ZSSSSM_GPU( char *storev, magma_int_t *m1, magma_int_t *n1, 
                        magma_int_t *m2, magma_int_t *n2, magma_int_t *k, magma_int_t *ib, 
                        devptr_t *dA1, magma_int_t *ldda1, 
                        devptr_t *dA2, magma_int_t *ldda2, 
                        devptr_t *dL1, magma_int_t *lddl1, 
                        devptr_t *dL2, magma_int_t *lddl2,
                        magma_int_t *IPIV, magma_int_t *info)
{
    magma_zssssm_gpu( storev[0], *m1, *n1,  *m2, *n2, *k, *ib,  
                      DEVPTR(dA1), *ldda1,  
                      DEVPTR(dA2), *ldda2,  
                      DEVPTR(dL1), *lddl1,  
                      DEVPTR(dL2), *lddl2,
                      IPIV, info);
}

void MAGMAF_ZUNGQR_GPU( magma_int_t *m, magma_int_t *n, magma_int_t *k, 
                        devptr_t *da, magma_int_t *ldda, 
                        cuDoubleComplex *tau, devptr_t *dwork, 
                        magma_int_t *nb, magma_int_t *info )
{
    magma_zungqr_gpu( *m, *n, *k,  
                      DEVPTR(da), *ldda, tau, 
                      DEVPTR(dwork), *nb, info );
}

void MAGMAF_ZUNMQR_GPU( char *side, char *trans, 
                        magma_int_t *m, magma_int_t *n, magma_int_t *k,
                        devptr_t *a,    magma_int_t *lda, cuDoubleComplex *tau, 
                        devptr_t *c,    magma_int_t *ldc,
                        devptr_t *work, magma_int_t *lwork, 
                        devptr_t *td,   magma_int_t *nb, magma_int_t *info)
{
    magma_zunmqr_gpu( side[0], trans[0], *m, *n, *k, 
                      DEVPTR(a),    *lda, tau,  
                      DEVPTR(c),    *ldc, 
                      DEVPTR(work), *lwork,  
                      DEVPTR(td),   *nb, info);
}

#ifdef __cplusplus
}
#endif
