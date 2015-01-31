/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c
       @author Mark Gates
*/

#ifndef MAGMA_ZGEHRD_H
#define MAGMA_ZGEHRD_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct zgehrd_data
{
    magma_int_t ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    magmaDoubleComplex *A    [ MagmaMaxGPUs ];  // ldda*nlocal
    magmaDoubleComplex *V    [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    magmaDoubleComplex *Vd   [ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    magmaDoubleComplex *Y    [ MagmaMaxGPUs ];  // ldda*nb
    magmaDoubleComplex *W    [ MagmaMaxGPUs ];  // ldda*nb
    magmaDoubleComplex *Ti   [ MagmaMaxGPUs ];  // nb*nb
    
    magma_queue_t streams[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_ZGEHRD_H
