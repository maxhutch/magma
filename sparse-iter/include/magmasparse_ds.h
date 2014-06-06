/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magmasparse_zc.h mixed zc -> ds, Fri May 30 10:41:30 2014
       @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_DS_H
#define MAGMASPARSE_DS_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_d


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
magma_vector_dlag2s(              magma_d_vector x, 
                                  magma_s_vector *y );

magma_int_t
magma_sparse_matrix_dlag2s(       magma_d_sparse_matrix A, 
                                  magma_s_sparse_matrix *B );


magma_int_t
magma_vector_slag2d(              magma_s_vector x, 
                                  magma_d_vector *y );

magma_int_t
magma_sparse_matrix_slag2d(       magma_s_sparse_matrix A, 
                                  magma_d_sparse_matrix *B );

void
magmablas_dlag2s_sparse(          magma_int_t M, magma_int_t N , 
                                  const double *A, magma_int_t lda, 
                                  float *SA, magma_int_t ldsa, 
                                  magma_int_t *info );

void 
magmablas_slag2d_sparse(          magma_int_t M, magma_int_t N , 
                                  const float *SA, magma_int_t ldsa, 
                                  double *A, magma_int_t lda, 
                                  magma_int_t *info );

void 
magma_dlag2s_CSR_DENSE(           magma_d_sparse_matrix A, 
                                  magma_s_sparse_matrix *B );

void 
magma_dlag2s_CSR_DENSE_alloc(     magma_d_sparse_matrix A, 
                                  magma_s_sparse_matrix *B );

void 
magma_dlag2s_CSR_DENSE_convert(   magma_d_sparse_matrix A, 
                                  magma_s_sparse_matrix *B );

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
magma_dspgmres(    magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
                   magma_d_solver_par *solver_par,  
                   magma_d_preconditioner *precond_par);

magma_int_t
magma_dspbicgstab( magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
                   magma_d_solver_par *solver_par,  
                   magma_d_preconditioner *precond_par);

magma_int_t
magma_dsir(        magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
                   magma_d_solver_par *solver_par, 
                   magma_d_preconditioner *precond_par );

magma_int_t
magma_dspir(       magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
                   magma_d_solver_par *solver_par, 
                   magma_d_preconditioner *precond_par );


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/



/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE BLAS function definitions
*/



#ifdef __cplusplus
}
#endif

#undef PRECISION_d
#endif /* MAGMASPARSE_DS_H */
