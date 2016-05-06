/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magma_zgehrd_m.h normal z -> d, Mon May  2 23:31:25 2016
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
    
    magma_queue_t queues[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_DGEHRD_H
