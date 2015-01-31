/**
 *
 * @file plasma_wrap.h
 *
 * This is a temporary file to compile imported PLASMA routines. Need to be removed later.
 *
 * @author Simplice Donfack
 * @date January 2015
 *
 *
 **/
#ifndef PLASMA_WRAP_H
#define PLASMA_WRAP_H

//#include "dbg_complex_to_double.h"
//#include "quark.h"

//typedef float  _Complex PLASMA_Complex32_t;
//typedef double _Complex
    //magmaDoubleComplex;





//#define QUARK_Get_RankInTask(quark) 1
/* CBLAS requires for scalar arguments to be passed by address rather than by value */
#ifndef CBLAS_SADDR
#define CBLAS_SADDR( _val_ ) &(_val_)
#endif


//#ifdef USE_MKL
//#include "mkl_lapacke.h"

//#endif



extern "C"  void CORE_zgetrf_reclap_init(void);
//extern "C"  void QUARK_CORE_zgetrf_reclap(Quark *quark, Quark_Task_Flags *task_flags,
//                              int m, int n, int nb,
//                              double *A, int lda,
//                              int *IPIV,
//                              int iinfo,
//                              int nbthread);

//void CORE_zgetrf_reclap_quark(Quark* quark);
//int CORE_zgetrf_reclap(int M, int N, double *A, int LDA, int *IPIV, int *info);
#endif
