/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magma_zgehrd_m.h, normal z -> s, Sun Nov 20 20:20:46 2016
       @author Mark Gates
*/

#ifndef MAGMA_SGEHRD_H
#define MAGMA_SGEHRD_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
    Structure containing matrices for multi-GPU sgehrd.

    - dA  is distributed column block-cyclic across GPUs.
    - dV  is duplicated on all GPUs.
    - dVd is distributed row block-cyclic across GPUs (TODO: verify).
    - dY  is partial results on each GPU in slahr2,
          then complete results are duplicated on all GPUs for slahru.
    - dW  is local to each GPU (workspace).
    - dTi is duplicated on all GPUs.

    @ingroup magma_gehrd
*******************************************************************************/
struct sgehrd_data
{
    magma_int_t ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    magmaFloat_ptr dA [ MagmaMaxGPUs ];  // ldda*nlocal
    magmaFloat_ptr dV [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    magmaFloat_ptr dVd[ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    magmaFloat_ptr dY [ MagmaMaxGPUs ];  // ldda*nb
    magmaFloat_ptr dW [ MagmaMaxGPUs ];  // ldda*nb
    magmaFloat_ptr dTi[ MagmaMaxGPUs ];  // nb*nb
    
    magma_queue_t queues[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_SGEHRD_H
