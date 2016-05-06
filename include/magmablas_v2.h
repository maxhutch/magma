/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
*/

#ifndef MAGMABLAS_V2_H
#define MAGMABLAS_V2_H

#include "magmablas_z_v2.h"
#include "magmablas_c_v2.h"
#include "magmablas_d_v2.h"
#include "magmablas_s_v2.h"
#include "magmablas_zc_v2.h"
#include "magmablas_ds_v2.h"

#ifdef __cplusplus
extern "C" {
#endif


// ========================================
// queue support
// new magma_queue_create adds device
#undef magma_queue_create

#define magma_queue_create  magma_queue_create_v2

// ========================================
// copying vectors
// set copies host to device
// get copies device to host
// async versions are same for v1 and v2; see magmablas_q.h

#undef magma_setvector
#undef magma_getvector
#undef magma_copyvector

#define magma_setvector    magma_setvector_q
#define magma_getvector    magma_getvector_q
#define magma_copyvector   magma_copyvector_q


// ========================================
// copying sub-matrices (contiguous columns)
// set  copies host to device
// get  copies device to host
// copy copies device to device

#undef magma_setmatrix
#undef magma_getmatrix
#undef magma_copymatrix

#define magma_setmatrix    magma_setmatrix_q
#define magma_getmatrix    magma_getmatrix_q
#define magma_copymatrix   magma_copymatrix_q


// ========================================
// copying vectors - version for magma_int_t

#undef magma_isetvector
#undef magma_igetvector
#undef magma_icopyvector

#define magma_isetvector    magma_isetvector_q
#define magma_igetvector    magma_igetvector_q
#define magma_icopyvector   magma_icopyvector_q


// ========================================
// copying sub-matrices - version for magma_int_t

#undef magma_isetmatrix
#undef magma_igetmatrix
#undef magma_icopymatrix

#define magma_isetmatrix    magma_isetmatrix_q
#define magma_igetmatrix    magma_igetmatrix_q
#define magma_icopymatrix   magma_icopymatrix_q


// ========================================
// copying vectors - version for magma_index_t

#undef magma_index_setvector
#undef magma_index_getvector
#undef magma_index_copyvector

#define magma_index_setvector    magma_index_setvector_q
#define magma_index_getvector    magma_index_getvector_q
#define magma_index_copyvector   magma_index_copyvector_q


// ========================================
// copying sub-matrices - version for magma_index_t

#undef magma_index_setmatrix
#undef magma_index_getmatrix
#undef magma_index_copymatrix

#define magma_index_setmatrix    magma_index_setmatrix_q
#define magma_index_getmatrix    magma_index_getmatrix_q
#define magma_index_copymatrix   magma_index_copymatrix_q


#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_V2_H */
