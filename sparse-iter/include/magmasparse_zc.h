/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions mixed zc -> ds
       @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_ZC_H
#define MAGMASPARSE_ZC_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Matrix Descriptors
*/


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/
magma_int_t
magma_vector_zlag2c(              magma_z_vector x, 
                                  magma_c_vector *y );

magma_int_t
magma_sparse_matrix_zlag2c(       magma_z_sparse_matrix A, 
                                  magma_c_sparse_matrix *B );


magma_int_t
magma_vector_clag2z(              magma_c_vector x, 
                                  magma_z_vector *y );

magma_int_t
magma_sparse_matrix_clag2z(       magma_c_sparse_matrix A, 
                                  magma_z_sparse_matrix *B );

void
magmablas_zlag2c_sparse(          magma_int_t M, magma_int_t N , 
                                  const magmaDoubleComplex *A, magma_int_t lda, 
                                  magmaFloatComplex *SA, magma_int_t ldsa, 
                                  magma_int_t *info );

void 
magmablas_clag2z_sparse(          magma_int_t M, magma_int_t N , 
                                  const magmaFloatComplex *SA, magma_int_t ldsa, 
                                  magmaDoubleComplex *A, magma_int_t lda, 
                                  magma_int_t *info );

void 
magma_zlag2c_CSR_DENSE(           magma_z_sparse_matrix A, 
                                  magma_c_sparse_matrix *B );

void 
magma_zlag2c_CSR_DENSE_alloc(     magma_z_sparse_matrix A, 
                                  magma_c_sparse_matrix *B );

void 
magma_zlag2c_CSR_DENSE_convert(   magma_z_sparse_matrix A, 
                                  magma_c_sparse_matrix *B );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU
*/


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on GPU
*/
magma_int_t
magma_zcpgmres(    magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                   magma_z_solver_par *solver_par,  
                   magma_z_preconditioner *precond_par);

magma_int_t
magma_zcpbicgstab( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                   magma_z_solver_par *solver_par,  
                   magma_z_preconditioner *precond_par);

magma_int_t
magma_zcir(        magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                   magma_z_solver_par *solver_par, 
                   magma_z_preconditioner *precond_par );

magma_int_t
magma_zcpir(       magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                   magma_z_solver_par *solver_par, 
                   magma_z_preconditioner *precond_par );


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/



/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE BLAS function definitions
*/



#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_ZC_H */
