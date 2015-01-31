/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zgehrd_m.h normal z -> d, Fri Jan 30 19:00:05 2015
       @author Mark Gates
*/

#ifndef MAGMA_DGEHRD_H
#define MAGMA_DGEHRD_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dgehrd_data
{
    magma_int_t ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    double *A    [ MagmaMaxGPUs ];  // ldda*nlocal
    double *V    [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    double *Vd   [ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    double *Y    [ MagmaMaxGPUs ];  // ldda*nb
    double *W    [ MagmaMaxGPUs ];  // ldda*nb
    double *Ti   [ MagmaMaxGPUs ];  // nb*nb
    
    magma_queue_t streams[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_DGEHRD_H
