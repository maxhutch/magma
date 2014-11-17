/*
 -- MAGMA (version 1.6.0) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date November 2014

 @generated from magmasparse_z.h normal z -> c, Sat Nov 15 19:54:20 2014
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_C_H
#define MAGMASPARSE_C_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_c


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t
magma_cparse_opts( 
    int argc, 
    char** argv, 
    magma_copts *opts, 
    int *matrices, 
    magma_queue_t queue );

magma_int_t 
read_c_csr_from_binary( 
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    const char * filename,
    magma_queue_t queue );

magma_int_t 
read_c_csr_from_mtx( 
    magma_storage_t *type, 
    magma_location_t *location,
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_c_csr_mtx( 
    magma_c_sparse_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_ccsrset( 
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col, 
    magmaFloatComplex *val,
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_ccsrget( 
    magma_c_sparse_matrix A,
    magma_int_t *m, 
    magma_int_t *n, 
    magma_index_t **row, 
    magma_index_t **col, 
    magmaFloatComplex **val,
    magma_queue_t queue );


magma_int_t 
magma_cvset( 
    magma_int_t m, 
    magma_int_t n, 
    magmaFloatComplex *val,
    magma_c_vector *v,
    magma_queue_t queue );

magma_int_t 
magma_cvget( 
    magma_c_vector v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaFloatComplex **val,
    magma_queue_t queue );

magma_int_t 
magma_cvset_gpu( 
    magma_int_t m, 
    magma_int_t n, 
    magmaFloatComplex_ptr val,
    magma_c_vector *v,
    magma_queue_t queue );

magma_int_t 
magma_cvget_gpu( 
    magma_c_vector v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaFloatComplex_ptr *val,
    magma_queue_t queue );


magma_int_t 
magma_c_csr_mtxsymm( 
    magma_c_sparse_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_c_csr_compressor( 
    magmaFloatComplex ** val, 
    magma_index_t ** row, 
    magma_index_t ** col, 
    magmaFloatComplex ** valn, 
    magma_index_t ** rown, 
    magma_index_t ** coln, 
    magma_int_t *n,
    magma_queue_t queue );

magma_int_t
magma_cmcsrcompressor( 
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cmcsrcompressor_gpu( 
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_c_mtranspose( 
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_cvtranspose( 
    magma_c_vector x,
    magma_c_vector *y,
    magma_queue_t queue );

magma_int_t 
magma_c_cucsrtranspose( 
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *B,
    magma_queue_t queue );

magma_int_t 
c_transpose_csr( 
    magma_int_t n_rows, 
    magma_int_t n_cols, 
    magma_int_t nnz,
    magmaFloatComplex *val, 
    magma_index_t *row, 
    magma_index_t *col, 
    magma_int_t *new_n_rows, 
    magma_int_t *new_n_cols, 
    magma_int_t *new_nnz, 
    magmaFloatComplex **new_val, 
    magma_index_t **new_row, 
    magma_index_t **new_col,
    magma_queue_t queue );

magma_int_t
magma_ccsrsplit( 
    magma_int_t bsize,
    magma_c_sparse_matrix A,
    magma_c_sparse_matrix *D,
    magma_c_sparse_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_cmscale( 
    magma_c_sparse_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t 
magma_cmdiff( 
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix B, 
 real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_cmdiagadd( 
    magma_c_sparse_matrix *A, 
    magmaFloatComplex add,
    magma_queue_t queue );

magma_int_t 
magma_cmsort( 
    magma_c_sparse_matrix *A,
    magma_queue_t queue );


magma_int_t
magma_csymbilu( 
    magma_c_sparse_matrix *A, 
    magma_int_t levels,
    magma_c_sparse_matrix *L,
    magma_c_sparse_matrix *U,
    magma_queue_t queue );


magma_int_t 
write_c_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
write_c_csrtomtx( 
    magma_c_sparse_matrix A,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
print_c_csr( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    magma_queue_t queue );

magma_int_t 
print_c_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
    magma_queue_t queue );


magma_int_t 
magma_c_mtranspose(
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *B,
    magma_queue_t queue );


magma_int_t 
magma_c_mtransfer(
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *B, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_c_vtransfer(
    magma_c_vector x, 
    magma_c_vector *y, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_c_mconvert(
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *B, 
    magma_storage_t old_format, 
    magma_storage_t new_format,
    magma_queue_t queue );


magma_int_t
magma_c_vinit(
    magma_c_vector *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magmaFloatComplex values,
    magma_queue_t queue );

magma_int_t
magma_c_vvisu(
    magma_c_vector x, 
    magma_int_t offset, 
    magma_int_t displaylength,
    magma_queue_t queue );

magma_int_t
magma_c_vread(
    magma_c_vector *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue );

magma_int_t
magma_c_vspread(
    magma_c_vector *x, 
    const char * filename,
    magma_queue_t queue );

magma_int_t
magma_c_mvisu(
    magma_c_sparse_matrix A,
    magma_queue_t queue );

magma_int_t 
magma_cdiameter(
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_crowentries(
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_c_mfree(
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_c_vfree(
    magma_c_vector *x,
    magma_queue_t queue );

magma_int_t
magma_cresidual(
    magma_c_sparse_matrix A, 
    magma_c_vector b, 
    magma_c_vector x, 
    float *res,
    magma_queue_t queue );

magma_int_t
magma_cmgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    magmaFloatComplex *diag_vals,
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cm_27stencil(
    magma_int_t n,
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cm_5stencil(
    magma_int_t n,
    magma_c_sparse_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_csolverinfo(
    magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_csolverinfo_init(
    magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ceigensolverinfo_init(
    magma_c_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_csolverinfo_free(
    magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU
*/




/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE solvers (Data on GPU)
*/

magma_int_t 
magma_ccg(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_ccg_res(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_ccg_merge(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_cgmres(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab(
    magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector *x, 
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_merge(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_merge2(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cpcg(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbpcg(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpbicgstab(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpgmres(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cjacobi(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbaiter(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_citerref(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cilu(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_cbcsrlu(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_cbcsrlutrf(
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *M,
    magma_int_t *ipiv, 
    magma_int_t version,
    magma_queue_t queue );

magma_int_t
magma_cbcsrlusv(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );



magma_int_t
magma_cilucg(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cilugmres(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_solver_par *solver_par,
    magma_queue_t queue ); 


magma_int_t
magma_clobpcg_shift(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magma_int_t shift,
    magmaFloatComplex_ptr x,
    magma_queue_t queue );

magma_int_t
magma_clobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    float *evalues, 
    magmaFloatComplex_ptr X,
    magmaFloatComplex_ptr R, 
    float *res,
    magma_queue_t queue );

magma_int_t
magma_clobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaFloatComplex_ptr X,
    magmaFloatComplex_ptr Y,
    magma_queue_t queue );


/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_clobpcg(
    magma_c_sparse_matrix A,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );




/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_cjacobisetup(
    magma_c_sparse_matrix A, 
    magma_c_vector b, 
    magma_c_sparse_matrix *M, 
    magma_c_vector *c,
    magma_queue_t queue );

magma_int_t
magma_cjacobisetup_matrix(
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix *M, 
    magma_c_vector *d,
    magma_queue_t queue );

magma_int_t
magma_cjacobisetup_vector(
    magma_c_vector b, 
    magma_c_vector d, 
    magma_c_vector *c,
    magma_queue_t queue );

magma_int_t
magma_cjacobiiter(
    magma_c_sparse_matrix M, 
    magma_c_vector c, 
    magma_c_vector *x, 
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cjacobiiter_precond( 
    magma_c_sparse_matrix M, 
    magma_c_vector *x, 
    magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_cpastixsetup(
    magma_c_sparse_matrix A, magma_c_vector b,
    magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_capplypastix(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


// CUSPARSE preconditioner

magma_int_t
magma_ccuilusetup(
    magma_c_sparse_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuilu_l(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuilu_r(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_ccuiccsetup(
    magma_c_sparse_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuicc_l(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuicc_r(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_ccumilusetup(
    magma_c_sparse_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumilu_l(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumilu_r(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_ccumiccsetup(
    magma_c_sparse_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumicc_l(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumicc_r(
    magma_c_vector b, magma_c_vector *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


// block-asynchronous iteration

magma_int_t
magma_cbajac_csr(
    magma_int_t localiters,
    magma_c_sparse_matrix D,
    magma_c_sparse_matrix R,
    magma_c_vector b,
    magma_c_vector *x,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_c_spmv(
    magmaFloatComplex alpha, 
    magma_c_sparse_matrix A, 
    magma_c_vector x, 
    magmaFloatComplex beta, 
    magma_c_vector y,
    magma_queue_t queue );

magma_int_t
magma_c_spmv_shift(
    magmaFloatComplex alpha, 
    magma_c_sparse_matrix A, 
    magmaFloatComplex lambda,
    magma_c_vector x, 
    magmaFloatComplex beta, 
    magma_int_t offset, 
    magma_int_t blocksize,
    magmaIndex_ptr dadd_vecs, 
    magma_c_vector y,
    magma_queue_t queue );

magma_int_t
magma_ccuspmm(
    magma_c_sparse_matrix A, 
    magma_c_sparse_matrix B, 
    magma_c_sparse_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_ccuspaxpy(
    magmaFloatComplex_ptr alpha, magma_c_sparse_matrix A, 
    magmaFloatComplex_ptr beta, magma_c_sparse_matrix B, 
    magma_c_sparse_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_c_precond(
    magma_c_sparse_matrix A, 
    magma_c_vector b, magma_c_vector *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_c_solver(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_copts *zopts,
    magma_queue_t queue );

magma_int_t
magma_c_precondsetup(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_c_applyprecond(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_c_applyprecond_left(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_c_applyprecond_right(
    magma_c_sparse_matrix A, magma_c_vector b, 
    magma_c_vector *x, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_c_initP2P(
    magma_int_t *bandwidth_benchmark,
    magma_int_t *num_gpus,
    magma_queue_t queue );

magma_int_t
magma_ccompact(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float *dnorms, float tol, 
    magma_int_t *activeMask, magma_int_t *cBlockSize,
    magma_queue_t queue );

magma_int_t
magma_ccompactActive(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magma_int_t *active,
    magma_queue_t queue );

magma_int_t
magma_cmlumerge(    
    magma_c_sparse_matrix L, 
    magma_c_sparse_matrix U,
    magma_c_sparse_matrix *A, 
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_cgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_cgecsrmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex lambda,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_cmgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_cgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_cgeellmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex lambda,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_cmgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_cgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_cgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex lambda,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_cmgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_cgeellrtmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowlength,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_int_t num_threads,
    magma_int_t threads_per_row,
    magma_queue_t queue );

magma_int_t 
magma_cgesellcmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_cgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_cmgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_cmgesellpmv_blocked(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue );


magma_int_t
magma_cmergedgs(
    magma_int_t n, 
    magma_int_t ldh,
    magma_int_t k, 
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_ccopyscale(    
    int n, 
    int k,
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dv,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_scnrm2scale(    
    int m, 
    magmaFloatComplex_ptr dr,    
    int lddr, 
    magmaFloatComplex *drnorm,
    magma_queue_t queue );


magma_int_t
magma_cjacobisetup_vector_gpu(
    int num_rows, 
    magma_c_vector b, 
    magma_c_vector d, 
    magma_c_vector c,
    magma_c_vector *x,
    magma_queue_t queue );


magma_int_t
magma_cjacobi_diagscal(    
    int num_rows, 
    magma_c_vector d, 
    magma_c_vector b, 
    magma_c_vector *c,
    magma_queue_t queue );

magma_int_t
magma_cjacobisetup_diagscal(
    magma_c_sparse_matrix A, magma_c_vector *d,
    magma_queue_t queue );


magma_int_t
magma_cbicgmerge1(    
    int n, 
    magmaFloatComplex_ptr dskp,
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dp,
    magma_queue_t queue );


magma_int_t
magma_cbicgmerge2(
    int n, 
    magmaFloatComplex_ptr dskp, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr ds,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge3(
    int n, 
    magmaFloatComplex_ptr dskp, 
    magmaFloatComplex_ptr dp,
    magmaFloatComplex_ptr ds,
    magmaFloatComplex_ptr dt,
    magmaFloatComplex_ptr dx, 
    magmaFloatComplex_ptr dr,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge4(
    int type, 
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_ccgmerge_spmv1( 
    magma_c_sparse_matrix A,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr dd,
    magmaFloatComplex_ptr dz,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_ccgmerge_xrbeta( 
    int n,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd,
    magmaFloatComplex_ptr dz, 
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cmdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cgemvmdot(
    int n, 
    int k, 
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge_spmv1( 
    magma_c_sparse_matrix A,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr dp,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dv,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge_spmv2( 
    magma_c_sparse_matrix A,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr ds,
    magmaFloatComplex_ptr dt,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge_xrbeta( 
    int n,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr drr,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dp,
    magmaFloatComplex_ptr ds,
    magmaFloatComplex_ptr dt,
    magmaFloatComplex_ptr dx, 
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cbcsrswp(
    magma_int_t n,
    magma_int_t size_b, 
    magma_int_t *ipiv,
    magmaFloatComplex_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_cbcsrtrsv(
    magma_uplo_t uplo,
    magma_int_t r_blocks,
    magma_int_t c_blocks,
    magma_int_t size_b, 
    magmaFloatComplex_ptr dA,
    magma_index_t *blockinfo, 
    magmaFloatComplex_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_cbcsrvalcpy(
    magma_int_t size_b, 
    magma_int_t num_blocks, 
    magma_int_t num_zero_blocks, 
    magmaFloatComplex_ptr *dAval, 
    magmaFloatComplex_ptr *dBval,
    magmaFloatComplex_ptr *dBval2,
    magma_queue_t queue );

magma_int_t
magma_cbcsrluegemm(
    magma_int_t size_b, 
    magma_int_t num_block_rows,
    magma_int_t kblocks,
    magmaFloatComplex_ptr *dA, 
    magmaFloatComplex_ptr *dB, 
    magmaFloatComplex_ptr *dC,
    magma_queue_t queue );

magma_int_t
magma_cbcsrlupivloc(
    magma_int_t size_b, 
    magma_int_t kblocks,
    magmaFloatComplex_ptr *dA, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_cbcsrblockinfo5(
    magma_int_t lustep,
    magma_int_t num_blocks, 
    magma_int_t c_blocks, 
    magma_int_t size_b,
    magma_index_t *blockinfo,
    magmaFloatComplex_ptr dval,
    magmaFloatComplex_ptr *AII,
    magma_queue_t queue );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_c
#endif /* MAGMASPARSE_C_H */
