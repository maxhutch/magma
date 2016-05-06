/*
 -- MAGMA (version 2.0.2) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date May 2016

 @generated from sparse-iter/include/magmasparse_z.h normal z -> c, Mon May  2 23:31:25 2016
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_C_H
#define MAGMASPARSE_C_H

#include "magma_types.h"
#include "magmasparse_types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_c


#ifdef __cplusplus
extern "C" {
#endif


/* ////////////////////////////////////////////////////////////////////////////
 -- For backwards compatability, map old (1.6.1) to new (1.6.2) function names
*/

#define magma_c_mtranspose  magma_cmtranspose
#define magma_c_mtransfer   magma_cmtransfer
#define magma_c_vtransfer   magma_cmtransfer
#define magma_c_mconvert    magma_cmconvert
#define magma_c_vinit       magma_cvinit
#define magma_c_vvisu       magma_cprint_vector
#define magma_c_vread       magma_cvread
#define magma_c_vspread     magma_cvspread
#define magma_c_mvisu       magma_cprint_matrix
#define magma_c_mfree       magma_cmfree
#define magma_c_vfree       magma_cmfree
#define write_c_csr_mtx     magma_cwrite_csr_mtx
#define write_c_csrtomtx    magma_cwrite_csrtomtx
#define print_c_csr         magma_cprint_csr_mtx


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
    magma_c_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_ccsrset( 
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col, 
    magmaFloatComplex *val,
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_ccsrget( 
    magma_c_matrix A,
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
    magma_c_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_cvget( 
    magma_c_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaFloatComplex **val,
    magma_queue_t queue );

magma_int_t 
magma_cvset_dev( 
    magma_int_t m, 
    magma_int_t n, 
    magmaFloatComplex_ptr val,
    magma_c_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_cvget_dev( 
    magma_c_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaFloatComplex_ptr *val,
    magma_queue_t queue );


magma_int_t 
magma_c_csr_mtxsymm( 
    magma_c_matrix *A, 
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
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cmshrink(
    magma_c_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_cmcsrcompressor_gpu( 
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cvtranspose( 
    magma_c_matrix x,
    magma_c_matrix *y,
    magma_queue_t queue );

magma_int_t 
magma_c_cucsrtranspose( 
    magma_c_matrix A, 
    magma_c_matrix *B,
    magma_queue_t queue );

magma_int_t
    magma_cmtransposeconjugate(
    magma_c_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_cmconjugate(
    magma_c_matrix *A,
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
    magma_int_t offset,
    magma_int_t bsize,
    magma_c_matrix A,
    magma_c_matrix *D,
    magma_c_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_cmscale( 
    magma_c_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t
magma_cmslice(
    magma_int_t num_slices,
    magma_int_t slice,
    magma_c_matrix A, 
    magma_c_matrix *B,
    magma_c_matrix *ALOC,
    magma_c_matrix *ANLOC,
    magma_index_t *comm_i,
    magmaFloatComplex *comm_v,
    magma_int_t *start,
    magma_int_t *end,
    magma_queue_t queue );

magma_int_t 
magma_cmdiff( 
    magma_c_matrix A, 
    magma_c_matrix B, 
 real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_cmdiagadd( 
    magma_c_matrix *A, 
    magmaFloatComplex add,
    magma_queue_t queue );

magma_int_t 
magma_cmsort(
    magmaFloatComplex *x,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_cindexsort(
    magma_index_t *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_csort(
    magmaFloatComplex *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_cindexsortval(
    magma_index_t *x,
    magmaFloatComplex *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_corderstatistics(
    magmaFloatComplex *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaFloatComplex *element,
    magma_queue_t queue );

magma_int_t
magma_cmorderstatistics(
    magmaFloatComplex *val,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaFloatComplex *element,
    magma_queue_t queue );

magma_int_t
magma_cdomainoverlap(
    magma_index_t num_rows,
    magma_int_t *num_indices,
    magma_index_t *rowptr,
    magma_index_t *colidx,
    magma_index_t *x,
    magma_queue_t queue );

magma_int_t
magma_csymbilu( 
    magma_c_matrix *A, 
    magma_int_t levels,
    magma_c_matrix *L,
    magma_c_matrix *U,
    magma_queue_t queue );


magma_int_t 
magma_cwrite_csr_mtx( 
    magma_c_matrix A,
    magma_order_t MajorType,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_cwrite_csrtomtx( 
    magma_c_matrix A,
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_cprint_csr( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    magma_queue_t queue );

magma_int_t 
magma_cprint_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaFloatComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
    magma_queue_t queue );


magma_int_t 
magma_cmtranspose(
    magma_c_matrix A, 
    magma_c_matrix *B,
    magma_queue_t queue );


magma_int_t 
magma_cmtransfer(
    magma_c_matrix A, 
    magma_c_matrix *B, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_cmconvert(
    magma_c_matrix A, 
    magma_c_matrix *B, 
    magma_storage_t old_format, 
    magma_storage_t new_format,
    magma_queue_t queue );


magma_int_t
magma_cvinit(
    magma_c_matrix *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magma_int_t num_cols,
    magmaFloatComplex values,
    magma_queue_t queue );

magma_int_t
magma_cprint_vector(
    magma_c_matrix x, 
    magma_int_t offset, 
    magma_int_t displaylength,
    magma_queue_t queue );

magma_int_t
magma_cvread(
    magma_c_matrix *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue );

magma_int_t
magma_cvspread(
    magma_c_matrix *x, 
    const char * filename,
    magma_queue_t queue );

magma_int_t
magma_cprint_matrix(
    magma_c_matrix A,
    magma_queue_t queue );

magma_int_t 
magma_cdiameter(
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_crowentries(
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cmfree(
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cresidual(
    magma_c_matrix A, 
    magma_c_matrix b, 
    magma_c_matrix x, 
    float *res,
    magma_queue_t queue );

magma_int_t
magma_cresidualvec(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_matrix x,
    magma_c_matrix *r,
    float *res,
    magma_queue_t queue );

magma_int_t
magma_cresidual_slice(
    magma_int_t start,
    magma_int_t end,
    magma_c_matrix A, 
    magma_c_matrix b, 
    magma_c_matrix x,
    float *res,
    magma_queue_t queue );

magma_int_t
magma_cmgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    magmaFloatComplex *diag_vals,
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cm_27stencil(
    magma_int_t n,
    magma_c_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_cm_5stencil(
    magma_int_t n,
    magma_c_matrix *A,
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
 -- MAGMA_SPARSE iterative incomplete factorizations
*/


magma_int_t
magma_citerilusetup( 
    magma_c_matrix A, 
    magma_c_matrix b,                                 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_citericsetup( 
    magma_c_matrix A, 
    magma_c_matrix b, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_citericupdate( 
    magma_c_matrix A, 
    magma_c_preconditioner *precond, 
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_capplyiteric_l( 
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplyiteric_r( 
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_citerilu_csr( 
    magma_c_matrix A,
    magma_c_matrix L,
    magma_c_matrix U,
    magma_queue_t queue );

magma_int_t
magma_citeriluupdate(
    magma_c_matrix A,
    magma_c_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_citeric_csr( 
    magma_c_matrix A,
    magma_c_matrix A_CSR,
    magma_queue_t queue );

magma_int_t 
magma_cfrobenius( 
    magma_c_matrix A, 
    magma_c_matrix B, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_cnonlinres(   
    magma_c_matrix A, 
    magma_c_matrix L,
    magma_c_matrix U, 
    magma_c_matrix *LU, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_cilures(   
    magma_c_matrix A, 
    magma_c_matrix L,
    magma_c_matrix U, 
    magma_c_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_cicres(       
    magma_c_matrix A, 
    magma_c_matrix C,
    magma_c_matrix CT, 
    magma_c_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_cinitguess( 
    magma_c_matrix A, 
    magma_c_matrix *L, 
    magma_c_matrix *U,
    magma_queue_t queue );

magma_int_t 
magma_cinitrecursiveLU( 
    magma_c_matrix A, 
    magma_c_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_cmLdiagadd( 
    magma_c_matrix *L,
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative dynamic ILU
*/
#ifdef _OPENMP

magma_int_t
magma_cmdynamicic_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_c_matrix *LU_new,
    magma_c_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue );

magma_int_t
magma_cmdynamicilu_set_thrs(
    magma_int_t num_rm,
    magma_c_matrix *LU,
    magmaFloatComplex *thrs,
    magma_queue_t queue );

magma_int_t
magma_cmdynamicilu_rm_thrs(
    magmaFloatComplex *thrs,
    magma_int_t *num_rm,
    magma_c_matrix *LU,    
    magma_c_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue );

magma_int_t
magma_cmdynamicic_sweep(
    magma_c_matrix A,
    magma_c_matrix *LU,
    magma_queue_t queue );

magma_int_t
magma_cmdynamicic_residuals(
    magma_c_matrix A,
    magma_c_matrix LU,
    magma_c_matrix *LU_new,
    magma_queue_t queue );

magma_int_t
magma_cmdynamicic_candidates(
    magma_c_matrix LU,
    magma_c_matrix *LU_new,
    magma_queue_t queue );

magma_int_t
magma_citerictsetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue );

#endif
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
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_ccg_res(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_ccg_merge(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cpcg_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_ccgs(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ccgs_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cpcgs(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpcgs_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cqmr(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cqmr_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ctfqmr(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_unrolled(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cptfqmr(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cptfqmr_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x, 
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicg(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_merge(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_merge2(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_merge3(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cpcg(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbpcg(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpbicg(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpbicgstab(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpbicgstab_merge(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cfgmres(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbfgmres(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cidr(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cidr_merge(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cidr_strms(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cpidr(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpidr_merge(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cpidr_strms(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbombard(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbombard_merge(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cjacobi(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cjacobidomainoverlap(
    magma_c_matrix A, 
    magma_c_matrix b, 
    magma_c_matrix *x,  
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cbaiter(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbaiter_overlap(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_citerref(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_cbcsrlu(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_cbcsrlutrf(
    magma_c_matrix A, 
    magma_c_matrix *M,
    magma_int_t *ipiv, 
    magma_int_t version,
    magma_queue_t queue );

magma_int_t
magma_cbcsrlusv(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );



magma_int_t
magma_cilucg(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cilugmres(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_solver_par *solver_par,
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
    magma_c_matrix A, 
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par, 
    magma_queue_t queue );

/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE LSQR (Data on GPU)
*/
magma_int_t
magma_clsqr(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue );

/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_cjacobisetup(
    magma_c_matrix A, 
    magma_c_matrix b, 
    magma_c_matrix *M, 
    magma_c_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_cjacobisetup_matrix(
    magma_c_matrix A, 
    magma_c_matrix *M, 
    magma_c_matrix *d,
    magma_queue_t queue );

magma_int_t
magma_cjacobisetup_vector(
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_cjacobiiter(
    magma_c_matrix M, 
    magma_c_matrix c, 
    magma_c_matrix *x, 
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_cjacobiiter_precond( 
    magma_c_matrix M, 
    magma_c_matrix *x, 
    magma_c_solver_par *solver_par, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_cjacobiiter_sys(
    magma_c_matrix A, 
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix t, 
    magma_c_matrix *x,  
    magma_c_solver_par *solver_par,
    magma_queue_t queue );

// custom preconditioner
magma_int_t
magma_capplycustomprecond_l(
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycustomprecond_r(
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


// CUSPARSE preconditioner

magma_int_t
magma_ccuilusetup(
    magma_c_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ccumilusetup_transpose(
    magma_c_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuilu_l(
    magma_c_matrix b, magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuilu_r(
    magma_c_matrix b, magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ccuiccsetup(
    magma_c_matrix A, magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuicc_l(
    magma_c_matrix b, magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycuicc_r(
    magma_c_matrix b, magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_ccumilusetup(
    magma_c_matrix A, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ccustomilusetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ccustomicsetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ccumilugeneratesolverinfo(
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumilu_l(
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumilu_r(
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumilu_l_transpose(
    magma_c_matrix b, magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumilu_r_transpose(
    magma_c_matrix b, magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );



magma_int_t
magma_ccumiccsetup(
    magma_c_matrix A, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ccumicgeneratesolverinfo(
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumicc_l(
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_capplycumicc_r(
    magma_c_matrix b, 
    magma_c_matrix *x, 
    magma_c_preconditioner *precond,
    magma_queue_t queue );


// block-asynchronous iteration

magma_int_t
magma_cbajac_csr(
    magma_int_t localiters,
    magma_c_matrix D,
    magma_c_matrix R,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_cbajac_csr_overlap(
    magma_int_t localiters,
    magma_int_t matrices,
    magma_int_t overlap,
    magma_c_matrix *D,
    magma_c_matrix *R,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_c_spmv(
    magmaFloatComplex alpha, 
    magma_c_matrix A, 
    magma_c_matrix x, 
    magmaFloatComplex beta, 
    magma_c_matrix y,
    magma_queue_t queue );

magma_int_t
magma_ccustomspmv(
    magmaFloatComplex alpha, 
    magma_c_matrix x, 
    magmaFloatComplex beta, 
    magma_c_matrix y,
    magma_queue_t queue );

magma_int_t
magma_c_spmv_shift(
    magmaFloatComplex alpha, 
    magma_c_matrix A, 
    magmaFloatComplex lambda,
    magma_c_matrix x, 
    magmaFloatComplex beta, 
    magma_int_t offset, 
    magma_int_t blocksize,
    magmaIndex_ptr dadd_vecs, 
    magma_c_matrix y,
    magma_queue_t queue );

magma_int_t
magma_ccuspmm(
    magma_c_matrix A, 
    magma_c_matrix B, 
    magma_c_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_c_spmm(
    magmaFloatComplex alpha, 
    magma_c_matrix A,
    magma_c_matrix B,
    magma_c_matrix *C,
    magma_queue_t queue );

magma_int_t
magma_csymbilu( 
    magma_c_matrix *A, 
    magma_int_t levels, 
    magma_c_matrix *L, 
    magma_c_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_ccuspaxpy(
    magmaFloatComplex_ptr alpha, magma_c_matrix A, 
    magmaFloatComplex_ptr beta, magma_c_matrix B, 
    magma_c_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_c_precond(
    magma_c_matrix A, 
    magma_c_matrix b, magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_c_solver(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_copts *zopts,
    magma_queue_t queue );

magma_int_t
magma_c_precondsetup(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_solver_par *solver,
    magma_c_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_c_applyprecond(
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_c_applyprecond_left(
    magma_trans_t trans,
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_c_applyprecond_right(
    magma_trans_t trans,
    magma_c_matrix A, magma_c_matrix b, 
    magma_c_matrix *x, magma_c_preconditioner *precond,
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
    magma_c_matrix L, 
    magma_c_matrix U,
    magma_c_matrix *A, 
    magma_queue_t queue );

magma_int_t
magma_cdiagcheck(
    magma_c_matrix dA,
    magma_queue_t queue );


/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE wrappers to dense MAGMA
*/
magma_int_t
magma_cqr(
    magma_int_t m, 
    magma_int_t n, 
    magma_c_matrix A, 
    magma_int_t lda, 
    magma_c_matrix *Q, 
    magma_c_matrix *R, 
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/

magma_int_t
magma_cgeaxpy(
    magmaFloatComplex alpha,
    magma_c_matrix X,
    magmaFloatComplex beta,
    magma_c_matrix *Y,
    magma_queue_t queue );

magma_int_t
magma_cgecsrreimsplit(
    magma_c_matrix A,
    magma_c_matrix *ReA,
    magma_c_matrix *ImA,
    magma_queue_t queue );

magma_int_t
magma_cgedensereimsplit(
    magma_c_matrix A,
    magma_c_matrix *ReA,
    magma_c_matrix *ImA,
    magma_queue_t queue );

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
    magma_int_t offset,
    magma_int_t blocksize,
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
    magma_int_t offset,
    magma_int_t blocksize,
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
    magma_int_t offset,
    magma_int_t blocksize,
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
    magma_int_t n, 
    magma_int_t k,
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dv,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_scnrm2scale(    
    magma_int_t m, 
    magmaFloatComplex_ptr dr,    
    magma_int_t lddr, 
    magmaFloatComplex *drnorm,
    magma_queue_t queue );


magma_int_t
magma_cjacobisetup_vector_gpu(
    magma_int_t num_rows, 
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix c,
    magma_c_matrix *x,
    magma_queue_t queue );


magma_int_t
magma_cjacobi_diagscal(    
    magma_int_t num_rows, 
    magma_c_matrix d, 
    magma_c_matrix b, 
    magma_c_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_cjacobiupdate(
    magma_c_matrix t, 
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_cjacobispmvupdate(
    magma_int_t maxiter,
    magma_c_matrix A, 
    magma_c_matrix t, 
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_cjacobispmvupdate_bw(
    magma_int_t maxiter,
    magma_c_matrix A, 
    magma_c_matrix t, 
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_cjacobispmvupdateselect(
    magma_int_t maxiter,
    magma_int_t num_updates,
    magma_index_t *indices,
    magma_c_matrix A,
    magma_c_matrix t, 
    magma_c_matrix b, 
    magma_c_matrix d, 
    magma_c_matrix tmp, 
    magma_c_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_cjacobisetup_diagscal(
    magma_c_matrix A, magma_c_matrix *d,
    magma_queue_t queue );


//##################   kernel fusion for Krylov methods

magma_int_t
magma_cmergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex_ptr alpha, 
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr x,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge1(    
    magma_int_t n, 
    magmaFloatComplex_ptr dskp,
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dp,
    magma_queue_t queue );


magma_int_t
magma_cbicgmerge2(
    magma_int_t n, 
    magmaFloatComplex_ptr dskp, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr ds,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge3(
    magma_int_t n, 
    magmaFloatComplex_ptr dskp, 
    magmaFloatComplex_ptr dp,
    magmaFloatComplex_ptr ds,
    magmaFloatComplex_ptr dt,
    magmaFloatComplex_ptr dx, 
    magmaFloatComplex_ptr dr,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge4(
    magma_int_t type, 
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cidr_smoothing_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex_ptr drs,
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dt, 
    magma_queue_t queue );

magma_int_t
magma_cidr_smoothing_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex omega,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dxs, 
    magma_queue_t queue );

magma_int_t
magma_ccgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr q, 
    magmaFloatComplex_ptr u,
    magmaFloatComplex_ptr p,
    magma_queue_t queue );

magma_int_t
magma_ccgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr u,
    magmaFloatComplex_ptr p, 
    magma_queue_t queue );

magma_int_t
magma_ccgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr v_hat,
    magmaFloatComplex_ptr u, 
    magmaFloatComplex_ptr q,
    magmaFloatComplex_ptr t, 
    magma_queue_t queue );

magma_int_t
magma_ccgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr u_hat,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r,
    magma_queue_t queue );

magma_int_t
magma_cqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex rho,
    magmaFloatComplex psi,
    magmaFloatComplex_ptr y, 
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr w,
    magma_queue_t queue );

magma_int_t
magma_cqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex pde,
    magmaFloatComplex rde,
    magmaFloatComplex_ptr y,
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr p, 
    magmaFloatComplex_ptr q, 
    magma_queue_t queue );

magma_int_t
magma_cqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr pt,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr y,
    magma_queue_t queue );

magma_int_t
magma_cqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex eta,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr pt,
    magmaFloatComplex_ptr d, 
    magmaFloatComplex_ptr s, 
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_cqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex eta,
    magmaFloatComplex pds,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr pt,
    magmaFloatComplex_ptr d, 
    magmaFloatComplex_ptr s, 
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_cqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex rho,
    magmaFloatComplex psi,
    magmaFloatComplex_ptr y, 
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr w,
    magmaFloatComplex_ptr wt,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex omega,
    magmaFloatComplex_ptr r, 
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr p,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr s, 
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex omega,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr s,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_ptr r,
    magma_queue_t queue );

magma_int_t
magma_cbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex omega,
    magmaFloatComplex_ptr y,
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr s,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_ptr r,
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex sigma,
    magmaFloatComplex_ptr v, 
    magmaFloatComplex_ptr Au,
    magmaFloatComplex_ptr u_m,
    magmaFloatComplex_ptr pu_m,
    magmaFloatComplex_ptr u_mp1,
    magmaFloatComplex_ptr w, 
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex eta,
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr w,
    magmaFloatComplex_ptr u_m,
    magmaFloatComplex_ptr u_mp1, 
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr Au_new,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr Au, 
    magma_queue_t queue );

magma_int_t
magma_ctfqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex sigma,
    magmaFloatComplex_ptr v, 
    magmaFloatComplex_ptr Au,
    magmaFloatComplex_ptr u_mp1,
    magmaFloatComplex_ptr w, 
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magma_queue_t queue );

magma_int_t
magma_ccgmerge_spmv1( 
    magma_c_matrix A,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr dd,
    magmaFloatComplex_ptr dz,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_ccgmerge_xrbeta( 
    magma_int_t n,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd,
    magmaFloatComplex_ptr dz, 
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_cpcgmerge_xrbeta1(
    magma_int_t n,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd,
    magmaFloatComplex_ptr dz, 
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cpcgmerge_xrbeta2(
    magma_int_t n,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr dh,
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dd, 
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cjcgmerge_xrbeta(
    magma_int_t n,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr diag,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd,
    magmaFloatComplex_ptr dz,
    magmaFloatComplex_ptr dh, 
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cmdotc_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
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
magma_cgemvmdot_shfl(
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
    magma_int_t n, 
    magma_int_t k, 
    magmaFloatComplex_ptr dv, 
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_cmdotc1(
    magma_int_t n,  
    magmaFloatComplex_ptr v0, 
    magmaFloatComplex_ptr w0,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cmdotc2(
    magma_int_t n,  
    magmaFloatComplex_ptr v0, 
    magmaFloatComplex_ptr w0,
    magmaFloatComplex_ptr v1, 
    magmaFloatComplex_ptr w1,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cmdotc3(
    magma_int_t n,  
    magmaFloatComplex_ptr v0, 
    magmaFloatComplex_ptr w0,
    magmaFloatComplex_ptr v1, 
    magmaFloatComplex_ptr w1,
    magmaFloatComplex_ptr v2, 
    magmaFloatComplex_ptr w2,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cmdotc4(
    magma_int_t n,  
    magmaFloatComplex_ptr v0, 
    magmaFloatComplex_ptr w0,
    magmaFloatComplex_ptr v1, 
    magmaFloatComplex_ptr w1,
    magmaFloatComplex_ptr v2, 
    magmaFloatComplex_ptr w2,
    magmaFloatComplex_ptr v3, 
    magmaFloatComplex_ptr w3,
    magmaFloatComplex_ptr d1,
    magmaFloatComplex_ptr d2,
    magmaFloatComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge_spmv1( 
    magma_c_matrix A,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr dp,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dv,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge_spmv2( 
    magma_c_matrix A,
    magmaFloatComplex_ptr dd1,
    magmaFloatComplex_ptr dd2,
    magmaFloatComplex_ptr ds,
    magmaFloatComplex_ptr dt,
    magmaFloatComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_cbicgmerge_xrbeta( 
    magma_int_t n,
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
