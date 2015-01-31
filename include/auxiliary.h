/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
*/

#ifndef MAGMA_AUXILIARY_H
#define MAGMA_AUXILIARY_H

#include "magma_types.h"

/* ------------------------------------------------------------
 *   -- MAGMA Auxiliary structures and functions
 * --------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

real_Double_t magma_wtime( void );
real_Double_t magma_sync_wtime( magma_queue_t queue );

size_t magma_strlcpy(char *dst, const char *src, size_t siz);

magma_int_t magma_num_gpus( void );

double magma_cabs(magmaDoubleComplex x);
float  magma_cabsf(magmaFloatComplex x);

magma_int_t magma_is_devptr( const void* A );

// magma GPU-complex PCIe connection
magma_int_t magma_buildconnection_mgpu(  magma_int_t gnode[MagmaMaxGPUs+2][MagmaMaxGPUs+2], magma_int_t *nbcmplx, magma_int_t ngpu);

void magma_indices_1D_bcyclic( magma_int_t nb, magma_int_t ngpu, magma_int_t dev,
                               magma_int_t j0, magma_int_t j1,
                               magma_int_t* dj0, magma_int_t* dj1 );

void magma_print_environment();

void swp2pswp(magma_trans_t trans, magma_int_t n, magma_int_t *ipiv, magma_int_t *newipiv);

#ifdef __cplusplus
}
#endif

#endif  // MAGMA_AUXILIARY_H
