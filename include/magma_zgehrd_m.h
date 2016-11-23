/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @precisions normal z -> s d c
       @author Mark Gates
*/

#ifndef MAGMA_ZGEHRD_H
#define MAGMA_ZGEHRD_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
    Structure containing matrices for multi-GPU zgehrd.

    - dA  is distributed column block-cyclic across GPUs.
    - dV  is duplicated on all GPUs.
    - dVd is distributed row block-cyclic across GPUs (TODO: verify).
    - dY  is partial results on each GPU in zlahr2,
          then complete results are duplicated on all GPUs for zlahru.
    - dW  is local to each GPU (workspace).
    - dTi is duplicated on all GPUs.

    @ingroup magma_gehrd
*******************************************************************************/
struct zgehrd_data
{
    magma_int_t ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    magmaDoubleComplex_ptr dA [ MagmaMaxGPUs ];  // ldda*nlocal
    magmaDoubleComplex_ptr dV [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    magmaDoubleComplex_ptr dVd[ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    magmaDoubleComplex_ptr dY [ MagmaMaxGPUs ];  // ldda*nb
    magmaDoubleComplex_ptr dW [ MagmaMaxGPUs ];  // ldda*nb
    magmaDoubleComplex_ptr dTi[ MagmaMaxGPUs ];  // nb*nb
    
    magma_queue_t queues[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_ZGEHRD_H
