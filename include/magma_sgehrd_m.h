/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from magma_zgehrd_m.h normal z -> s, Fri Sep 11 18:29:17 2015
       @author Mark Gates
*/

#ifndef MAGMA_SGEHRD_H
#define MAGMA_SGEHRD_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sgehrd_data
{
    magma_int_t ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    float *A    [ MagmaMaxGPUs ];  // ldda*nlocal
    float *V    [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    float *Vd   [ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    float *Y    [ MagmaMaxGPUs ];  // ldda*nb
    float *W    [ MagmaMaxGPUs ];  // ldda*nb
    float *Ti   [ MagmaMaxGPUs ];  // nb*nb
    
    magma_queue_t streams[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_SGEHRD_H
