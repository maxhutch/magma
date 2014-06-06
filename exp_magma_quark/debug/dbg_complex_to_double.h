/*This file enable conversion from complex to double for debug,
 * that is automatically done when compiling MAGMA, so this file do not need to be redistribute.
 *
 *@author: Simplice Donfack
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
