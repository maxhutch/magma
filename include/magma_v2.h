/*
    MAGMA (version 2.1.0)
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    @date August 2016
*/

#ifndef MAGMA_V2_H
#define MAGMA_V2_H

/* ------------------------------------------------------------
 * MAGMA BLAS Functions
 * --------------------------------------------------------- */
#include "magmablas_v2.h"
#include "magmablas_q.h"
#include "magma_batched.h"
#include "magma_vbatched.h"
#include "magma_bulge.h"

/* ------------------------------------------------------------
 * MAGMA functions
 * --------------------------------------------------------- */
#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"
#include "magma_zc.h"
#include "magma_ds.h"
#include "magma_auxiliary.h"

#ifdef __cplusplus
extern "C" {
#endif

// this can be used instead of magma_init() if NO v1 interfaces are ever called.
magma_int_t magma_init_v2( void );

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_V2_H
