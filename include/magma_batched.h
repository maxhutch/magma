/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
*/

#ifndef MAGMABATCHED_H
#define MAGMABATCHED_H
#include "magma_types.h"

/* ------------------------------------------------------------
 * MAGMA BATCHED functions
 * --------------------------------------------------------- */
#include "magma_zbatched.h"
#include "magma_cbatched.h"
#include "magma_dbatched.h"
#include "magma_sbatched.h"



#ifdef __cplusplus
extern "C" {
#endif


void setup_pivinfo_batched( magma_int_t **pivinfo_array, magma_int_t **ipiv_array, 
                              magma_int_t m, magma_int_t nb, 
                              magma_int_t batchCount,  magma_queue_t queue);


void adjust_ipiv_batched( magma_int_t **ipiv_array, 
                       magma_int_t m, magma_int_t offset, 
                       magma_int_t batchCount, magma_queue_t queue);

void magma_idisplace_pointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue);

void stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount, magma_queue_t queue);

void set_ipointer(magma_int_t **output_array,
                 magma_int_t *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column, 
                 magma_int_t batchSize,
                 magma_int_t batchCount, magma_queue_t queue);

#ifdef __cplusplus
}
#endif


#endif /* MAGMABATCHED_H */
