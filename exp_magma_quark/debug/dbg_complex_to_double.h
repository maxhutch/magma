/*
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 */
/* This temporary file convert complex function to double for debug only.
 * Magma handle conversion from complex to double during the compilation, so there is no need to distribute this file.
 */
#define COMPLEX_TO_DOUBLE /*Direction for complex*/
#ifdef COMPLEX_TO_DOUBLE

#ifndef DBG_COMPLEX_WRAP_H
#define DBG_COMPLEX_WRAP_H

#define PLASMA_Complex64_t double /*Compilation only*/

/*Some MKL function that need a translation*/
#define cblas_izamax(a,b,c) cblas_idamax((a),(b),(c))
#define cblas_ztrsm  cblas_dtrsm
#define cblas_zgemm  cblas_dgemm
#define cblas_zscal  cblas_dscal
#define cabs(a) ((a)>=0?(a):-(a))
//#define cabs abs 
#else

#endif

#endif

