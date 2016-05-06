/*
 -- MAGMA (version 2.0.2) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date May 2016

 @generated from sparse-iter/include/magmasparse_z.h normal z -> d, Mon May  2 23:31:25 2016
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_D_H
#define MAGMASPARSE_D_H

#include "magma_types.h"
#include "magmasparse_types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_d


#ifdef __cplusplus
extern "C" {
#endif


/* ////////////////////////////////////////////////////////////////////////////
 -- For backwards compatability, map old (1.6.1) to new (1.6.2) function names
*/

#define magma_d_mtranspose  magma_dmtranspose
#define magma_d_mtransfer   magma_dmtransfer
#define magma_d_vtransfer   magma_dmtransfer
#define magma_d_mconvert    magma_dmconvert
#define magma_d_vinit       magma_dvinit
#define magma_d_vvisu       magma_dprint_vector
#define magma_d_vread       magma_dvread
#define magma_d_vspread     magma_dvspread
#define magma_d_mvisu       magma_dprint_matrix
#define magma_d_mfree       magma_dmfree
#define magma_d_vfree       magma_dmfree
#define write_d_csr_mtx     magma_dwrite_csr_mtx
#define write_d_csrtomtx    magma_dwrite_csrtomtx
#define print_d_csr         magma_dprint_csr_mtx


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t
magma_dparse_opts( 
    int argc, 
    char** argv, 
    magma_dopts *opts, 
    int *matrices, 
    magma_queue_t queue );

magma_int_t 
read_d_csr_from_binary( 
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    double **val, 
    magma_index_t **row, 
    magma_index_t **col,
    const char * filename,
    magma_queue_t queue );

magma_int_t 
read_d_csr_from_mtx( 
    magma_storage_t *type, 
    magma_location_t *location,
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    double **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_d_csr_mtx( 
    magma_d_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_dcsrset( 
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col, 
    double *val,
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_dcsrget( 
    magma_d_matrix A,
    magma_int_t *m, 
    magma_int_t *n, 
    magma_index_t **row, 
    magma_index_t **col, 
    double **val,
    magma_queue_t queue );


magma_int_t 
magma_dvset( 
    magma_int_t m, 
    magma_int_t n, 
    double *val,
    magma_d_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_dvget( 
    magma_d_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    double **val,
    magma_queue_t queue );

magma_int_t 
magma_dvset_dev( 
    magma_int_t m, 
    magma_int_t n, 
    magmaDouble_ptr val,
    magma_d_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_dvget_dev( 
    magma_d_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDouble_ptr *val,
    magma_queue_t queue );


magma_int_t 
magma_d_csr_mtxsymm( 
    magma_d_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_d_csr_compressor( 
    double ** val, 
    magma_index_t ** row, 
    magma_index_t ** col, 
    double ** valn, 
    magma_index_t ** rown, 
    magma_index_t ** coln, 
    magma_int_t *n,
    magma_queue_t queue );

magma_int_t
magma_dmcsrcompressor( 
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dmshrink(
    magma_d_matrix A,
    magma_d_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_dmcsrcompressor_gpu( 
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dvtranspose( 
    magma_d_matrix x,
    magma_d_matrix *y,
    magma_queue_t queue );

magma_int_t 
magma_d_cucsrtranspose( 
    magma_d_matrix A, 
    magma_d_matrix *B,
    magma_queue_t queue );

magma_int_t
    magma_dmtransposeconjugate(
    magma_d_matrix A,
    magma_d_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_dmconjugate(
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t 
d_transpose_csr( 
    magma_int_t n_rows, 
    magma_int_t n_cols, 
    magma_int_t nnz,
    double *val, 
    magma_index_t *row, 
    magma_index_t *col, 
    magma_int_t *new_n_rows, 
    magma_int_t *new_n_cols, 
    magma_int_t *new_nnz, 
    double **new_val, 
    magma_index_t **new_row, 
    magma_index_t **new_col,
    magma_queue_t queue );

magma_int_t
magma_dcsrsplit( 
    magma_int_t offset,
    magma_int_t bsize,
    magma_d_matrix A,
    magma_d_matrix *D,
    magma_d_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_dmscale( 
    magma_d_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t
magma_dmslice(
    magma_int_t num_slices,
    magma_int_t slice,
    magma_d_matrix A, 
    magma_d_matrix *B,
    magma_d_matrix *ALOC,
    magma_d_matrix *ANLOC,
    magma_index_t *comm_i,
    double *comm_v,
    magma_int_t *start,
    magma_int_t *end,
    magma_queue_t queue );

magma_int_t 
magma_dmdiff( 
    magma_d_matrix A, 
    magma_d_matrix B, 
 real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_dmdiagadd( 
    magma_d_matrix *A, 
    double add,
    magma_queue_t queue );

magma_int_t 
magma_dmsort(
    double *x,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_dindexsort(
    magma_index_t *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_dsort(
    double *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_dindexsortval(
    magma_index_t *x,
    double *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_dorderstatistics(
    double *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    double *element,
    magma_queue_t queue );

magma_int_t
magma_dmorderstatistics(
    double *val,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    double *element,
    magma_queue_t queue );

magma_int_t
magma_ddomainoverlap(
    magma_index_t num_rows,
    magma_int_t *num_indices,
    magma_index_t *rowptr,
    magma_index_t *colidx,
    magma_index_t *x,
    magma_queue_t queue );

magma_int_t
magma_dsymbilu( 
    magma_d_matrix *A, 
    magma_int_t levels,
    magma_d_matrix *L,
    magma_d_matrix *U,
    magma_queue_t queue );


magma_int_t 
magma_dwrite_csr_mtx( 
    magma_d_matrix A,
    magma_order_t MajorType,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_dwrite_csrtomtx( 
    magma_d_matrix A,
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_dprint_csr( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    double **val, 
    magma_index_t **row, 
    magma_index_t **col,
    magma_queue_t queue );

magma_int_t 
magma_dprint_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    double **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
    magma_queue_t queue );


magma_int_t 
magma_dmtranspose(
    magma_d_matrix A, 
    magma_d_matrix *B,
    magma_queue_t queue );


magma_int_t 
magma_dmtransfer(
    magma_d_matrix A, 
    magma_d_matrix *B, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_dmconvert(
    magma_d_matrix A, 
    magma_d_matrix *B, 
    magma_storage_t old_format, 
    magma_storage_t new_format,
    magma_queue_t queue );


magma_int_t
magma_dvinit(
    magma_d_matrix *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magma_int_t num_cols,
    double values,
    magma_queue_t queue );

magma_int_t
magma_dprint_vector(
    magma_d_matrix x, 
    magma_int_t offset, 
    magma_int_t displaylength,
    magma_queue_t queue );

magma_int_t
magma_dvread(
    magma_d_matrix *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue );

magma_int_t
magma_dvspread(
    magma_d_matrix *x, 
    const char * filename,
    magma_queue_t queue );

magma_int_t
magma_dprint_matrix(
    magma_d_matrix A,
    magma_queue_t queue );

magma_int_t 
magma_ddiameter(
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_drowentries(
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dmfree(
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dresidual(
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_matrix x, 
    double *res,
    magma_queue_t queue );

magma_int_t
magma_dresidualvec(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_matrix x,
    magma_d_matrix *r,
    double *res,
    magma_queue_t queue );

magma_int_t
magma_dresidual_slice(
    magma_int_t start,
    magma_int_t end,
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_matrix x,
    double *res,
    magma_queue_t queue );

magma_int_t
magma_dmgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    double *diag_vals,
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dm_27stencil(
    magma_int_t n,
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dm_5stencil(
    magma_int_t n,
    magma_d_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_dsolverinfo(
    magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dsolverinfo_init(
    magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_deigensolverinfo_init(
    magma_d_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_dsolverinfo_free(
    magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative incomplete factorizations
*/


magma_int_t
magma_diterilusetup( 
    magma_d_matrix A, 
    magma_d_matrix b,                                 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ditericsetup( 
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_ditericupdate( 
    magma_d_matrix A, 
    magma_d_preconditioner *precond, 
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_dapplyiteric_l( 
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplyiteric_r( 
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_diterilu_csr( 
    magma_d_matrix A,
    magma_d_matrix L,
    magma_d_matrix U,
    magma_queue_t queue );

magma_int_t
magma_diteriluupdate(
    magma_d_matrix A,
    magma_d_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_diteric_csr( 
    magma_d_matrix A,
    magma_d_matrix A_CSR,
    magma_queue_t queue );

magma_int_t 
magma_dfrobenius( 
    magma_d_matrix A, 
    magma_d_matrix B, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_dnonlinres(   
    magma_d_matrix A, 
    magma_d_matrix L,
    magma_d_matrix U, 
    magma_d_matrix *LU, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_dilures(   
    magma_d_matrix A, 
    magma_d_matrix L,
    magma_d_matrix U, 
    magma_d_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_dicres(       
    magma_d_matrix A, 
    magma_d_matrix C,
    magma_d_matrix CT, 
    magma_d_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_dinitguess( 
    magma_d_matrix A, 
    magma_d_matrix *L, 
    magma_d_matrix *U,
    magma_queue_t queue );

magma_int_t 
magma_dinitrecursiveLU( 
    magma_d_matrix A, 
    magma_d_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_dmLdiagadd( 
    magma_d_matrix *L,
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative dynamic ILU
*/
#ifdef _OPENMP

magma_int_t
magma_dmdynamicic_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_d_matrix *LU_new,
    magma_d_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue );

magma_int_t
magma_dmdynamicilu_set_thrs(
    magma_int_t num_rm,
    magma_d_matrix *LU,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_dmdynamicilu_rm_thrs(
    double *thrs,
    magma_int_t *num_rm,
    magma_d_matrix *LU,    
    magma_d_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue );

magma_int_t
magma_dmdynamicic_sweep(
    magma_d_matrix A,
    magma_d_matrix *LU,
    magma_queue_t queue );

magma_int_t
magma_dmdynamicic_residuals(
    magma_d_matrix A,
    magma_d_matrix LU,
    magma_d_matrix *LU_new,
    magma_queue_t queue );

magma_int_t
magma_dmdynamicic_candidates(
    magma_d_matrix LU,
    magma_d_matrix *LU_new,
    magma_queue_t queue );

magma_int_t
magma_diterictsetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
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
magma_dcg(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_dcg_res(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_dcg_merge(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dpcg_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dcgs(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dcgs_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dpcgs(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dpcgs_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dqmr(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dqmr_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dtfqmr(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_unrolled(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dptfqmr(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dptfqmr_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x, 
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dbicg(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_merge(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_merge2(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_merge3(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dpcg(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dbpcg(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dpbicg(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dpbicgstab(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dpbicgstab_merge(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dfgmres(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dbfgmres(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_didr(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_didr_merge(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_didr_strms(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dpidr(
    magma_d_matrix A, magma_d_matrix b,
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dpidr_merge(
    magma_d_matrix A, magma_d_matrix b,
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dpidr_strms(
    magma_d_matrix A, magma_d_matrix b,
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dbombard(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dbombard_merge(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_djacobi(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_djacobidomainoverlap(
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_matrix *x,  
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dbaiter(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dbaiter_overlap(
    magma_d_matrix A, magma_d_matrix b,
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_diterref(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_dbcsrlu(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_dbcsrlutrf(
    magma_d_matrix A, 
    magma_d_matrix *M,
    magma_int_t *ipiv, 
    magma_int_t version,
    magma_queue_t queue );

magma_int_t
magma_dbcsrlusv(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );



magma_int_t
magma_dilucg(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_dilugmres(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue ); 


magma_int_t
magma_dlobpcg_shift(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magma_int_t shift,
    magmaDouble_ptr x,
    magma_queue_t queue );

magma_int_t
magma_dlobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    double *evalues, 
    magmaDouble_ptr X,
    magmaDouble_ptr R, 
    double *res,
    magma_queue_t queue );

magma_int_t
magma_dlobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaDouble_ptr X,
    magmaDouble_ptr Y,
    magma_queue_t queue );




/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_dlobpcg(
    magma_d_matrix A, 
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par, 
    magma_queue_t queue );

/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE LSQR (Data on GPU)
*/
magma_int_t
magma_dlsqr(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue );

/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_djacobisetup(
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_matrix *M, 
    magma_d_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_djacobisetup_matrix(
    magma_d_matrix A, 
    magma_d_matrix *M, 
    magma_d_matrix *d,
    magma_queue_t queue );

magma_int_t
magma_djacobisetup_vector(
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_djacobiiter(
    magma_d_matrix M, 
    magma_d_matrix c, 
    magma_d_matrix *x, 
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_djacobiiter_precond( 
    magma_d_matrix M, 
    magma_d_matrix *x, 
    magma_d_solver_par *solver_par, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_djacobiiter_sys(
    magma_d_matrix A, 
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix t, 
    magma_d_matrix *x,  
    magma_d_solver_par *solver_par,
    magma_queue_t queue );

// custom preconditioner
magma_int_t
magma_dapplycustomprecond_l(
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycustomprecond_r(
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );


// CUSPARSE preconditioner

magma_int_t
magma_dcuilusetup(
    magma_d_matrix A, magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dcumilusetup_transpose(
    magma_d_matrix A, magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycuilu_l(
    magma_d_matrix b, magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycuilu_r(
    magma_d_matrix b, magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dcuiccsetup(
    magma_d_matrix A, magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycuicc_l(
    magma_d_matrix b, magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycuicc_r(
    magma_d_matrix b, magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_dcumilusetup(
    magma_d_matrix A, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dcustomilusetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dcustomicsetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dcumilugeneratesolverinfo(
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycumilu_l(
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycumilu_r(
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycumilu_l_transpose(
    magma_d_matrix b, magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycumilu_r_transpose(
    magma_d_matrix b, magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );



magma_int_t
magma_dcumiccsetup(
    magma_d_matrix A, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dcumicgeneratesolverinfo(
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycumicc_l(
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_dapplycumicc_r(
    magma_d_matrix b, 
    magma_d_matrix *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue );


// block-asynchronous iteration

magma_int_t
magma_dbajac_csr(
    magma_int_t localiters,
    magma_d_matrix D,
    magma_d_matrix R,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_dbajac_csr_overlap(
    magma_int_t localiters,
    magma_int_t matrices,
    magma_int_t overlap,
    magma_d_matrix *D,
    magma_d_matrix *R,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_d_spmv(
    double alpha, 
    magma_d_matrix A, 
    magma_d_matrix x, 
    double beta, 
    magma_d_matrix y,
    magma_queue_t queue );

magma_int_t
magma_dcustomspmv(
    double alpha, 
    magma_d_matrix x, 
    double beta, 
    magma_d_matrix y,
    magma_queue_t queue );

magma_int_t
magma_d_spmv_shift(
    double alpha, 
    magma_d_matrix A, 
    double lambda,
    magma_d_matrix x, 
    double beta, 
    magma_int_t offset, 
    magma_int_t blocksize,
    magmaIndex_ptr dadd_vecs, 
    magma_d_matrix y,
    magma_queue_t queue );

magma_int_t
magma_dcuspmm(
    magma_d_matrix A, 
    magma_d_matrix B, 
    magma_d_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_d_spmm(
    double alpha, 
    magma_d_matrix A,
    magma_d_matrix B,
    magma_d_matrix *C,
    magma_queue_t queue );

magma_int_t
magma_dsymbilu( 
    magma_d_matrix *A, 
    magma_int_t levels, 
    magma_d_matrix *L, 
    magma_d_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_dcuspaxpy(
    magmaDouble_ptr alpha, magma_d_matrix A, 
    magmaDouble_ptr beta, magma_d_matrix B, 
    magma_d_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_d_precond(
    magma_d_matrix A, 
    magma_d_matrix b, magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_d_solver(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_dopts *zopts,
    magma_queue_t queue );

magma_int_t
magma_d_precondsetup(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_solver_par *solver,
    magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_d_applyprecond(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_d_applyprecond_left(
    magma_trans_t trans,
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_d_applyprecond_right(
    magma_trans_t trans,
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_d_initP2P(
    magma_int_t *bandwidth_benchmark,
    magma_int_t *num_gpus,
    magma_queue_t queue );

magma_int_t
magma_dcompact(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *dnorms, double tol, 
    magma_int_t *activeMask, magma_int_t *cBlockSize,
    magma_queue_t queue );

magma_int_t
magma_dcompactActive(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, 
    magma_int_t *active,
    magma_queue_t queue );

magma_int_t
magma_dmlumerge(    
    magma_d_matrix L, 
    magma_d_matrix U,
    magma_d_matrix *A, 
    magma_queue_t queue );

magma_int_t
magma_ddiagcheck(
    magma_d_matrix dA,
    magma_queue_t queue );


/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE wrappers to dense MAGMA
*/
magma_int_t
magma_dqr(
    magma_int_t m, 
    magma_int_t n, 
    magma_d_matrix A, 
    magma_int_t lda, 
    magma_d_matrix *Q, 
    magma_d_matrix *R, 
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/

magma_int_t
magma_dgeaxpy(
    double alpha,
    magma_d_matrix X,
    double beta,
    magma_d_matrix *Y,
    magma_queue_t queue );

magma_int_t
magma_dgecsrreimsplit(
    magma_d_matrix A,
    magma_d_matrix *ReA,
    magma_d_matrix *ImA,
    magma_queue_t queue );

magma_int_t
magma_dgedensereimsplit(
    magma_d_matrix A,
    magma_d_matrix *ReA,
    magma_d_matrix *ImA,
    magma_queue_t queue );

magma_int_t 
magma_dgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_dgecsrmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    double lambda,
    magmaDouble_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_dmgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_dgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_dgeellmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    double lambda,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDouble_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_dmgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_dgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_dgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    double lambda,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDouble_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_dmgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_dgeellrtmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowlength,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_int_t num_threads,
    magma_int_t threads_per_row,
    magma_queue_t queue );

magma_int_t 
magma_dgesellcmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_dgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_dmgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_dmgesellpmv_blocked(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue );


magma_int_t
magma_dmergedgs(
    magma_int_t n, 
    magma_int_t ldh,
    magma_int_t k, 
    magmaDouble_ptr dv, 
    magmaDouble_ptr dr,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dcopyscale(    
    magma_int_t n, 
    magma_int_t k,
    magmaDouble_ptr dr, 
    magmaDouble_ptr dv,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dnrm2scale(    
    magma_int_t m, 
    magmaDouble_ptr dr,    
    magma_int_t lddr, 
    double *drnorm,
    magma_queue_t queue );


magma_int_t
magma_djacobisetup_vector_gpu(
    magma_int_t num_rows, 
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix c,
    magma_d_matrix *x,
    magma_queue_t queue );


magma_int_t
magma_djacobi_diagscal(    
    magma_int_t num_rows, 
    magma_d_matrix d, 
    magma_d_matrix b, 
    magma_d_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_djacobiupdate(
    magma_d_matrix t, 
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_djacobispmvupdate(
    magma_int_t maxiter,
    magma_d_matrix A, 
    magma_d_matrix t, 
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_djacobispmvupdate_bw(
    magma_int_t maxiter,
    magma_d_matrix A, 
    magma_d_matrix t, 
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_djacobispmvupdateselect(
    magma_int_t maxiter,
    magma_int_t num_updates,
    magma_index_t *indices,
    magma_d_matrix A,
    magma_d_matrix t, 
    magma_d_matrix b, 
    magma_d_matrix d, 
    magma_d_matrix tmp, 
    magma_d_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_djacobisetup_diagscal(
    magma_d_matrix A, magma_d_matrix *d,
    magma_queue_t queue );


//##################   kernel fusion for Krylov methods

magma_int_t
magma_dmergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDouble_ptr alpha, 
    magmaDouble_ptr p,
    magmaDouble_ptr x,
    magma_queue_t queue );

magma_int_t
magma_dbicgmerge1(    
    magma_int_t n, 
    magmaDouble_ptr dskp,
    magmaDouble_ptr dv, 
    magmaDouble_ptr dr, 
    magmaDouble_ptr dp,
    magma_queue_t queue );


magma_int_t
magma_dbicgmerge2(
    magma_int_t n, 
    magmaDouble_ptr dskp, 
    magmaDouble_ptr dr,
    magmaDouble_ptr dv, 
    magmaDouble_ptr ds,
    magma_queue_t queue );

magma_int_t
magma_dbicgmerge3(
    magma_int_t n, 
    magmaDouble_ptr dskp, 
    magmaDouble_ptr dp,
    magmaDouble_ptr ds,
    magmaDouble_ptr dt,
    magmaDouble_ptr dx, 
    magmaDouble_ptr dr,
    magma_queue_t queue );

magma_int_t
magma_dbicgmerge4(
    magma_int_t type, 
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_didr_smoothing_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDouble_ptr drs,
    magmaDouble_ptr dr, 
    magmaDouble_ptr dt, 
    magma_queue_t queue );

magma_int_t
magma_didr_smoothing_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double omega,
    magmaDouble_ptr dx,
    magmaDouble_ptr dxs, 
    magma_queue_t queue );

magma_int_t
magma_dcgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr r,
    magmaDouble_ptr q, 
    magmaDouble_ptr u,
    magmaDouble_ptr p,
    magma_queue_t queue );

magma_int_t
magma_dcgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDouble_ptr r,
    magmaDouble_ptr u,
    magmaDouble_ptr p, 
    magma_queue_t queue );

magma_int_t
magma_dcgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    magmaDouble_ptr v_hat,
    magmaDouble_ptr u, 
    magmaDouble_ptr q,
    magmaDouble_ptr t, 
    magma_queue_t queue );

magma_int_t
magma_dcgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    magmaDouble_ptr u_hat,
    magmaDouble_ptr t,
    magmaDouble_ptr x, 
    magmaDouble_ptr r,
    magma_queue_t queue );

magma_int_t
magma_dqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double rho,
    double psi,
    magmaDouble_ptr y, 
    magmaDouble_ptr z,
    magmaDouble_ptr v,
    magmaDouble_ptr w,
    magma_queue_t queue );

magma_int_t
magma_dqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double pde,
    double rde,
    magmaDouble_ptr y,
    magmaDouble_ptr z,
    magmaDouble_ptr p, 
    magmaDouble_ptr q, 
    magma_queue_t queue );

magma_int_t
magma_dqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr pt,
    magmaDouble_ptr v,
    magmaDouble_ptr y,
    magma_queue_t queue );

magma_int_t
magma_dqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double eta,
    magmaDouble_ptr p,
    magmaDouble_ptr pt,
    magmaDouble_ptr d, 
    magmaDouble_ptr s, 
    magmaDouble_ptr x, 
    magmaDouble_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_dqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double eta,
    double pds,
    magmaDouble_ptr p,
    magmaDouble_ptr pt,
    magmaDouble_ptr d, 
    magmaDouble_ptr s, 
    magmaDouble_ptr x, 
    magmaDouble_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_dqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    double rho,
    double psi,
    magmaDouble_ptr y, 
    magmaDouble_ptr z,
    magmaDouble_ptr v,
    magmaDouble_ptr w,
    magmaDouble_ptr wt,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    double omega,
    magmaDouble_ptr r, 
    magmaDouble_ptr v,
    magmaDouble_ptr p,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    magmaDouble_ptr r,
    magmaDouble_ptr v,
    magmaDouble_ptr s, 
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double omega,
    magmaDouble_ptr p,
    magmaDouble_ptr s,
    magmaDouble_ptr t,
    magmaDouble_ptr x,
    magmaDouble_ptr r,
    magma_queue_t queue );

magma_int_t
magma_dbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double omega,
    magmaDouble_ptr y,
    magmaDouble_ptr z,
    magmaDouble_ptr s,
    magmaDouble_ptr t,
    magmaDouble_ptr x,
    magmaDouble_ptr r,
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double sigma,
    magmaDouble_ptr v, 
    magmaDouble_ptr Au,
    magmaDouble_ptr u_m,
    magmaDouble_ptr pu_m,
    magmaDouble_ptr u_mp1,
    magmaDouble_ptr w, 
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double eta,
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magmaDouble_ptr x, 
    magmaDouble_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr w,
    magmaDouble_ptr u_m,
    magmaDouble_ptr u_mp1, 
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr Au_new,
    magmaDouble_ptr v,
    magmaDouble_ptr Au, 
    magma_queue_t queue );

magma_int_t
magma_dtfqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double sigma,
    magmaDouble_ptr v, 
    magmaDouble_ptr Au,
    magmaDouble_ptr u_mp1,
    magmaDouble_ptr w, 
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magma_queue_t queue );

magma_int_t
magma_dcgmerge_spmv1( 
    magma_d_matrix A,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr dd,
    magmaDouble_ptr dz,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dcgmerge_xrbeta( 
    magma_int_t n,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr dx,
    magmaDouble_ptr dr,
    magmaDouble_ptr dd,
    magmaDouble_ptr dz, 
    magmaDouble_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_dpcgmerge_xrbeta1(
    magma_int_t n,
    magmaDouble_ptr dx,
    magmaDouble_ptr dr,
    magmaDouble_ptr dd,
    magmaDouble_ptr dz, 
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_dpcgmerge_xrbeta2(
    magma_int_t n,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr dh,
    magmaDouble_ptr dr, 
    magmaDouble_ptr dd, 
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_djcgmerge_xrbeta(
    magma_int_t n,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr diag,
    magmaDouble_ptr dx,
    magmaDouble_ptr dr,
    magmaDouble_ptr dd,
    magmaDouble_ptr dz,
    magmaDouble_ptr dh, 
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_dmdotc_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDouble_ptr dv, 
    magmaDouble_ptr dr,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dmdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaDouble_ptr dv, 
    magmaDouble_ptr dr,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dgemvmdot_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDouble_ptr dv, 
    magmaDouble_ptr dr,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_dgemvmdot(
    magma_int_t n, 
    magma_int_t k, 
    magmaDouble_ptr dv, 
    magmaDouble_ptr dr,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_dmdotc1(
    magma_int_t n,  
    magmaDouble_ptr v0, 
    magmaDouble_ptr w0,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_dmdotc2(
    magma_int_t n,  
    magmaDouble_ptr v0, 
    magmaDouble_ptr w0,
    magmaDouble_ptr v1, 
    magmaDouble_ptr w1,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_dmdotc3(
    magma_int_t n,  
    magmaDouble_ptr v0, 
    magmaDouble_ptr w0,
    magmaDouble_ptr v1, 
    magmaDouble_ptr w1,
    magmaDouble_ptr v2, 
    magmaDouble_ptr w2,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_dmdotc4(
    magma_int_t n,  
    magmaDouble_ptr v0, 
    magmaDouble_ptr w0,
    magmaDouble_ptr v1, 
    magmaDouble_ptr w1,
    magmaDouble_ptr v2, 
    magmaDouble_ptr w2,
    magmaDouble_ptr v3, 
    magmaDouble_ptr w3,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_dbicgmerge_spmv1( 
    magma_d_matrix A,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr dp,
    magmaDouble_ptr dr,
    magmaDouble_ptr dv,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dbicgmerge_spmv2( 
    magma_d_matrix A,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr ds,
    magmaDouble_ptr dt,
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dbicgmerge_xrbeta( 
    magma_int_t n,
    magmaDouble_ptr dd1,
    magmaDouble_ptr dd2,
    magmaDouble_ptr drr,
    magmaDouble_ptr dr,
    magmaDouble_ptr dp,
    magmaDouble_ptr ds,
    magmaDouble_ptr dt,
    magmaDouble_ptr dx, 
    magmaDouble_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dbcsrswp(
    magma_int_t n,
    magma_int_t size_b, 
    magma_int_t *ipiv,
    magmaDouble_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_dbcsrtrsv(
    magma_uplo_t uplo,
    magma_int_t r_blocks,
    magma_int_t c_blocks,
    magma_int_t size_b, 
    magmaDouble_ptr dA,
    magma_index_t *blockinfo, 
    magmaDouble_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_dbcsrvalcpy(
    magma_int_t size_b, 
    magma_int_t num_blocks, 
    magma_int_t num_zero_blocks, 
    magmaDouble_ptr *dAval, 
    magmaDouble_ptr *dBval,
    magmaDouble_ptr *dBval2,
    magma_queue_t queue );

magma_int_t
magma_dbcsrluegemm(
    magma_int_t size_b, 
    magma_int_t num_block_rows,
    magma_int_t kblocks,
    magmaDouble_ptr *dA, 
    magmaDouble_ptr *dB, 
    magmaDouble_ptr *dC,
    magma_queue_t queue );

magma_int_t
magma_dbcsrlupivloc(
    magma_int_t size_b, 
    magma_int_t kblocks,
    magmaDouble_ptr *dA, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_dbcsrblockinfo5(
    magma_int_t lustep,
    magma_int_t num_blocks, 
    magma_int_t c_blocks, 
    magma_int_t size_b,
    magma_index_t *blockinfo,
    magmaDouble_ptr dval,
    magmaDouble_ptr *AII,
    magma_queue_t queue );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_d
#endif /* MAGMASPARSE_D_H */
