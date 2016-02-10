/*
    -- MAGMA (version 2.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2016
*/

#ifndef MAGMA_H
#define MAGMA_H

// magma v1 includes cublas.h by default, unless cublas_v2.h has already been included
#ifndef CUBLAS_V2_H_
#include <cublas.h>
#endif

/* ------------------------------------------------------------
 * MAGMA BLAS Functions
 * --------------------------------------------------------- */
#include "magmablas_v1.h"
#include "magmablas_q.h"
#include "magma_batched.h"
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
#include "auxiliary.h"

#endif        //  #ifndef MAGMA_H
