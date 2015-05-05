/*
 -- MAGMA (version 1.1) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date

 @generated from magmasparse_z.h normal z -> s, Mon May  4 17:11:25 2015
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_S_H
#define MAGMASPARSE_S_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_s


#ifdef __cplusplus
extern "C" {
#endif


/* ////////////////////////////////////////////////////////////////////////////
 -- For backwards compatability, map old (1.6.1) to new (1.6.2) function names
*/

#define magma_s_mtranspose  magma_smtranspose
#define magma_s_mtransfer   magma_smtransfer
#define magma_s_vtransfer   magma_smtransfer
#define magma_s_mconvert    magma_smconvert
#define magma_s_vinit       magma_svinit
#define magma_s_vvisu       magma_sprint_vector
#define magma_s_vread       magma_svread
#define magma_s_vspread     magma_svspread
#define magma_s_mvisu       magma_sprint_matrix
#define magma_s_mfree       magma_smfree
#define magma_s_vfree       magma_smfree
#define write_s_csr_mtx     magma_swrite_csr_mtx
#define write_s_csrtomtx    magma_swrite_csrtomtx
#define print_s_csr         magma_sprint_csr_mtx


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t
magma_sparse_opts( 
    int argc, 
    char** argv, 
    magma_sopts *opts, 
    int *matrices, 
    magma_queue_t queue );

magma_int_t 
read_s_csr_from_binary( 
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    float **val, 
    magma_index_t **row, 
    magma_index_t **col,
    const char * filename,
    magma_queue_t queue );

magma_int_t 
read_s_csr_from_mtx( 
    magma_storage_t *type, 
    magma_location_t *location,
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    float **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_s_csr_mtx( 
    magma_s_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_scsrset( 
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col, 
    float *val,
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_scsrget( 
    magma_s_matrix A,
    magma_int_t *m, 
    magma_int_t *n, 
    magma_index_t **row, 
    magma_index_t **col, 
    float **val,
    magma_queue_t queue );


magma_int_t 
magma_svset( 
    magma_int_t m, 
    magma_int_t n, 
    float *val,
    magma_s_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_svget( 
    magma_s_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    float **val,
    magma_queue_t queue );

magma_int_t 
magma_svset_dev( 
    magma_int_t m, 
    magma_int_t n, 
    magmaFloat_ptr val,
    magma_s_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_svget_dev( 
    magma_s_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaFloat_ptr *val,
    magma_queue_t queue );


magma_int_t 
magma_s_csr_mtxsymm( 
    magma_s_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_s_csr_compressor( 
    float ** val, 
    magma_index_t ** row, 
    magma_index_t ** col, 
    float ** valn, 
    magma_index_t ** rown, 
    magma_index_t ** coln, 
    magma_int_t *n,
    magma_queue_t queue );

magma_int_t
magma_smcsrcompressor( 
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_smcsrcompressor_gpu( 
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_svtranspose( 
    magma_s_matrix x,
    magma_s_matrix *y,
    magma_queue_t queue );

magma_int_t 
magma_s_cucsrtranspose( 
    magma_s_matrix A, 
    magma_s_matrix *B,
    magma_queue_t queue );

magma_int_t 
s_transpose_csr( 
    magma_int_t n_rows, 
    magma_int_t n_cols, 
    magma_int_t nnz,
    float *val, 
    magma_index_t *row, 
    magma_index_t *col, 
    magma_int_t *new_n_rows, 
    magma_int_t *new_n_cols, 
    magma_int_t *new_nnz, 
    float **new_val, 
    magma_index_t **new_row, 
    magma_index_t **new_col,
    magma_queue_t queue );

magma_int_t
magma_scsrsplit( 
    magma_int_t bsize,
    magma_s_matrix A,
    magma_s_matrix *D,
    magma_s_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_smscale( 
    magma_s_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t 
magma_smdiff( 
    magma_s_matrix A, 
    magma_s_matrix B, 
 real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_smdiagadd( 
    magma_s_matrix *A, 
    float add,
    magma_queue_t queue );

magma_int_t 
magma_smsort( 
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_sindexsort(
    magma_index_t *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_sdomainoverlap(
    magma_index_t num_rows,
    magma_index_t *num_indices,
    magma_index_t *rowptr,
    magma_index_t *colidx,
    magma_index_t *x,
    magma_queue_t queue );

magma_int_t
magma_ssymbilu( 
    magma_s_matrix *A, 
    magma_int_t levels,
    magma_s_matrix *L,
    magma_s_matrix *U,
    magma_queue_t queue );


magma_int_t 
magma_swrite_csr_mtx( 
    magma_s_matrix A,
    magma_order_t MajorType,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_swrite_csrtomtx( 
    magma_s_matrix A,
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_sprint_csr( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    float **val, 
    magma_index_t **row, 
    magma_index_t **col,
    magma_queue_t queue );

magma_int_t 
magma_sprint_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    float **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
    magma_queue_t queue );


magma_int_t 
magma_smtranspose(
    magma_s_matrix A, 
    magma_s_matrix *B,
    magma_queue_t queue );


magma_int_t 
magma_smtransfer(
    magma_s_matrix A, 
    magma_s_matrix *B, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_smconvert(
    magma_s_matrix A, 
    magma_s_matrix *B, 
    magma_storage_t old_format, 
    magma_storage_t new_format,
    magma_queue_t queue );


magma_int_t
magma_svinit(
    magma_s_matrix *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magma_int_t num_cols,
    float values,
    magma_queue_t queue );

magma_int_t
magma_sprint_vector(
    magma_s_matrix x, 
    magma_int_t offset, 
    magma_int_t displaylength,
    magma_queue_t queue );

magma_int_t
magma_svread(
    magma_s_matrix *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue );

magma_int_t
magma_svspread(
    magma_s_matrix *x, 
    const char * filename,
    magma_queue_t queue );

magma_int_t
magma_sprint_matrix(
    magma_s_matrix A,
    magma_queue_t queue );

magma_int_t 
magma_sdiameter(
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_srowentries(
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_smfree(
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_sresidual(
    magma_s_matrix A, 
    magma_s_matrix b, 
    magma_s_matrix x, 
    float *res,
    magma_queue_t queue );

magma_int_t
magma_sresidualvec(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix x,
    magma_s_matrix *r,
    float *res,
    magma_queue_t queue );

magma_int_t
magma_smgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    float *diag_vals,
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_sm_27stencil(
    magma_int_t n,
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_sm_5stencil(
    magma_int_t n,
    magma_s_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_ssolverinfo(
    magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_ssolverinfo_init(
    magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_seigensolverinfo_init(
    magma_s_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_ssolverinfo_free(
    magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative incomplete factorizations
*/


magma_int_t
magma_siterilusetup( 
    magma_s_matrix A, 
    magma_s_matrix b,                                 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sitericsetup( 
    magma_s_matrix A, 
    magma_s_matrix b, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sitericupdate( 
    magma_s_matrix A, 
    magma_s_preconditioner *precond, 
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_sapplyiteric_l( 
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplyiteric_r( 
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_siterilu_csr( 
    magma_s_matrix A,
    magma_s_matrix L,
    magma_s_matrix U,
    magma_queue_t queue );

magma_int_t
magma_siteric_csr( 
    magma_s_matrix A,
    magma_s_matrix A_CSR,
    magma_queue_t queue );

magma_int_t 
magma_sfrobenius( 
    magma_s_matrix A, 
    magma_s_matrix B, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_snonlinres(   
    magma_s_matrix A, 
    magma_s_matrix L,
    magma_s_matrix U, 
    magma_s_matrix *LU, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_silures(   
    magma_s_matrix A, 
    magma_s_matrix L,
    magma_s_matrix U, 
    magma_s_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_sicres(       
    magma_s_matrix A, 
    magma_s_matrix C,
    magma_s_matrix CT, 
    magma_s_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_sinitguess( 
    magma_s_matrix A, 
    magma_s_matrix *L, 
    magma_s_matrix *U,
    magma_queue_t queue );

magma_int_t 
magma_sinitrecursiveLU( 
    magma_s_matrix A, 
    magma_s_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_smLdiagadd( 
    magma_s_matrix *L,
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
magma_scg(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_scg_res(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_scg_merge(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_sgmres(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_sbicgstab(
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x, 
    magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_sbicgstab_merge(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_sbicgstab_merge2(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_spcg(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_sbpcg(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_spbicgstab(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_spgmres(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_sfgmres(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_sjacobi(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_sjacobidomainoverlap(
    magma_s_matrix A, 
    magma_s_matrix b, 
    magma_s_matrix *x,  
    magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_sbaiter(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_siterref(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_silu(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_sbcsrlu(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_sbcsrlutrf(
    magma_s_matrix A, 
    magma_s_matrix *M,
    magma_int_t *ipiv, 
    magma_int_t version,
    magma_queue_t queue );

magma_int_t
magma_sbcsrlusv(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );



magma_int_t
magma_silucg(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_silugmres(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_solver_par *solver_par,
    magma_queue_t queue ); 


magma_int_t
magma_slobpcg_shift(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magma_int_t shift,
    magmaFloat_ptr x,
    magma_queue_t queue );

magma_int_t
magma_slobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    float *evalues, 
    magmaFloat_ptr X,
    magmaFloat_ptr R, 
    float *res,
    magma_queue_t queue );

magma_int_t
magma_slobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaFloat_ptr X,
    magmaFloat_ptr Y,
    magma_queue_t queue );


/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_slobpcg(
    magma_s_matrix A, 
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par, 
    magma_queue_t queue );




/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_sjacobisetup(
    magma_s_matrix A, 
    magma_s_matrix b, 
    magma_s_matrix *M, 
    magma_s_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_sjacobisetup_matrix(
    magma_s_matrix A, 
    magma_s_matrix *M, 
    magma_s_matrix *d,
    magma_queue_t queue );

magma_int_t
magma_sjacobisetup_vector(
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_sjacobiiter(
    magma_s_matrix M, 
    magma_s_matrix c, 
    magma_s_matrix *x, 
    magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_sjacobiiter_precond( 
    magma_s_matrix M, 
    magma_s_matrix *x, 
    magma_s_solver_par *solver_par, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sjacobiiter_sys(
    magma_s_matrix A, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix t, 
    magma_s_matrix *x,  
    magma_s_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_spastixsetup(
    magma_s_matrix A, magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_sapplypastix(
    magma_s_matrix b, magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


// custom preconditioner
magma_int_t
magma_sapplycustomprecond_l(
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycustomprecond_r(
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


// CUSPARSE preconditioner

magma_int_t
magma_scuilusetup(
    magma_s_matrix A, magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycuilu_l(
    magma_s_matrix b, magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycuilu_r(
    magma_s_matrix b, magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_scuiccsetup(
    magma_s_matrix A, magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycuicc_l(
    magma_s_matrix b, magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycuicc_r(
    magma_s_matrix b, magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_scumilusetup(
    magma_s_matrix A, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_scumilugeneratesolverinfo(
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycumilu_l(
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycumilu_r(
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_scumiccsetup(
    magma_s_matrix A, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_scumicgeneratesolverinfo(
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycumicc_l(
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_sapplycumicc_r(
    magma_s_matrix b, 
    magma_s_matrix *x, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );


// block-asynchronous iteration

magma_int_t
magma_sbajac_csr(
    magma_int_t localiters,
    magma_s_matrix D,
    magma_s_matrix R,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_s_spmv(
    float alpha, 
    magma_s_matrix A, 
    magma_s_matrix x, 
    float beta, 
    magma_s_matrix y,
    magma_queue_t queue );

magma_int_t
magma_scustomspmv(
    float alpha, 
    magma_s_matrix x, 
    float beta, 
    magma_s_matrix y,
    magma_queue_t queue );

magma_int_t
magma_s_spmv_shift(
    float alpha, 
    magma_s_matrix A, 
    float lambda,
    magma_s_matrix x, 
    float beta, 
    magma_int_t offset, 
    magma_int_t blocksize,
    magmaIndex_ptr dadd_vecs, 
    magma_s_matrix y,
    magma_queue_t queue );

magma_int_t
magma_scuspmm(
    magma_s_matrix A, 
    magma_s_matrix B, 
    magma_s_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_s_spmm(
    float alpha, 
    magma_s_matrix A,
    magma_s_matrix B,
    magma_s_matrix *C,
    magma_queue_t queue );

magma_int_t
magma_ssymbilu( 
    magma_s_matrix *A, 
    magma_int_t levels, 
    magma_s_matrix *L, 
    magma_s_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_scuspaxpy(
    magmaFloat_ptr alpha, magma_s_matrix A, 
    magmaFloat_ptr beta, magma_s_matrix B, 
    magma_s_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_s_precond(
    magma_s_matrix A, 
    magma_s_matrix b, magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_s_solver(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_sopts *zopts,
    magma_queue_t queue );

magma_int_t
magma_s_precondsetup(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_s_applyprecond(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_s_applyprecond_left(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_s_applyprecond_right(
    magma_s_matrix A, magma_s_matrix b, 
    magma_s_matrix *x, magma_s_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_s_initP2P(
    magma_int_t *bandwidth_benchmark,
    magma_int_t *num_gpus,
    magma_queue_t queue );

magma_int_t
magma_scompact(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *dnorms, float tol, 
    magma_int_t *activeMask, magma_int_t *cBlockSize,
    magma_queue_t queue );

magma_int_t
magma_scompactActive(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, 
    magma_int_t *active,
    magma_queue_t queue );

magma_int_t
magma_smlumerge(    
    magma_s_matrix L, 
    magma_s_matrix U,
    magma_s_matrix *A, 
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_sgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_sgecsrmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    float lambda,
    magmaFloat_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_smgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_sgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_sgeellmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    float alpha,
    float lambda,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaFloat_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_smgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_sgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_sgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    float alpha,
    float lambda,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaFloat_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_smgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_sgeellrtmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowlength,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_int_t num_threads,
    magma_int_t threads_per_row,
    magma_queue_t queue );

magma_int_t 
magma_sgesellcmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_sgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_smgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_smgesellpmv_blocked(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue );


magma_int_t
magma_smergedgs(
    magma_int_t n, 
    magma_int_t ldh,
    magma_int_t k, 
    magmaFloat_ptr dv, 
    magmaFloat_ptr dr,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_scopyscale(    
    int n, 
    int k,
    magmaFloat_ptr dr, 
    magmaFloat_ptr dv,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_snrm2scale(    
    int m, 
    magmaFloat_ptr dr,    
    int lddr, 
    float *drnorm,
    magma_queue_t queue );


magma_int_t
magma_sjacobisetup_vector_gpu(
    int num_rows, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix c,
    magma_s_matrix *x,
    magma_queue_t queue );


magma_int_t
magma_sjacobi_diagscal(    
    int num_rows, 
    magma_s_matrix d, 
    magma_s_matrix b, 
    magma_s_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_sjacobiupdate(
    magma_s_matrix t, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_sjacobispmvupdate(
    magma_int_t maxiter,
    magma_s_matrix A, 
    magma_s_matrix t, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_sjacobispmvupdate_bw(
    magma_int_t maxiter,
    magma_s_matrix A, 
    magma_s_matrix t, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_sjacobispmvupdateselect(
    magma_int_t maxiter,
    magma_int_t num_updates,
    magma_index_t *indices,
    magma_s_matrix A,
    magma_s_matrix t, 
    magma_s_matrix b, 
    magma_s_matrix d, 
    magma_s_matrix tmp, 
    magma_s_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_sjacobisetup_diagscal(
    magma_s_matrix A, magma_s_matrix *d,
    magma_queue_t queue );


magma_int_t
magma_sbicgmerge1(    
    int n, 
    magmaFloat_ptr dskp,
    magmaFloat_ptr dv, 
    magmaFloat_ptr dr, 
    magmaFloat_ptr dp,
    magma_queue_t queue );


magma_int_t
magma_sbicgmerge2(
    int n, 
    magmaFloat_ptr dskp, 
    magmaFloat_ptr dr,
    magmaFloat_ptr dv, 
    magmaFloat_ptr ds,
    magma_queue_t queue );

magma_int_t
magma_sbicgmerge3(
    int n, 
    magmaFloat_ptr dskp, 
    magmaFloat_ptr dp,
    magmaFloat_ptr ds,
    magmaFloat_ptr dt,
    magmaFloat_ptr dx, 
    magmaFloat_ptr dr,
    magma_queue_t queue );

magma_int_t
magma_sbicgmerge4(
    int type, 
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_scgmerge_spmv1( 
    magma_s_matrix A,
    magmaFloat_ptr d1,
    magmaFloat_ptr d2,
    magmaFloat_ptr dd,
    magmaFloat_ptr dz,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_scgmerge_xrbeta( 
    int n,
    magmaFloat_ptr d1,
    magmaFloat_ptr d2,
    magmaFloat_ptr dx,
    magmaFloat_ptr dr,
    magmaFloat_ptr dd,
    magmaFloat_ptr dz, 
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_smdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaFloat_ptr dv, 
    magmaFloat_ptr dr,
    magmaFloat_ptr dd1,
    magmaFloat_ptr dd2,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_sgemvmdot(
    int n, 
    int k, 
    magmaFloat_ptr dv, 
    magmaFloat_ptr dr,
    magmaFloat_ptr dd1,
    magmaFloat_ptr dd2,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_sbicgmerge_spmv1( 
    magma_s_matrix A,
    magmaFloat_ptr dd1,
    magmaFloat_ptr dd2,
    magmaFloat_ptr dp,
    magmaFloat_ptr dr,
    magmaFloat_ptr dv,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_sbicgmerge_spmv2( 
    magma_s_matrix A,
    magmaFloat_ptr dd1,
    magmaFloat_ptr dd2,
    magmaFloat_ptr ds,
    magmaFloat_ptr dt,
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_sbicgmerge_xrbeta( 
    int n,
    magmaFloat_ptr dd1,
    magmaFloat_ptr dd2,
    magmaFloat_ptr drr,
    magmaFloat_ptr dr,
    magmaFloat_ptr dp,
    magmaFloat_ptr ds,
    magmaFloat_ptr dt,
    magmaFloat_ptr dx, 
    magmaFloat_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_sbcsrswp(
    magma_int_t n,
    magma_int_t size_b, 
    magma_int_t *ipiv,
    magmaFloat_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_sbcsrtrsv(
    magma_uplo_t uplo,
    magma_int_t r_blocks,
    magma_int_t c_blocks,
    magma_int_t size_b, 
    magmaFloat_ptr dA,
    magma_index_t *blockinfo, 
    magmaFloat_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_sbcsrvalcpy(
    magma_int_t size_b, 
    magma_int_t num_blocks, 
    magma_int_t num_zero_blocks, 
    magmaFloat_ptr *dAval, 
    magmaFloat_ptr *dBval,
    magmaFloat_ptr *dBval2,
    magma_queue_t queue );

magma_int_t
magma_sbcsrluegemm(
    magma_int_t size_b, 
    magma_int_t num_block_rows,
    magma_int_t kblocks,
    magmaFloat_ptr *dA, 
    magmaFloat_ptr *dB, 
    magmaFloat_ptr *dC,
    magma_queue_t queue );

magma_int_t
magma_sbcsrlupivloc(
    magma_int_t size_b, 
    magma_int_t kblocks,
    magmaFloat_ptr *dA, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_sbcsrblockinfo5(
    magma_int_t lustep,
    magma_int_t num_blocks, 
    magma_int_t c_blocks, 
    magma_int_t size_b,
    magma_index_t *blockinfo,
    magmaFloat_ptr dval,
    magmaFloat_ptr *AII,
    magma_queue_t queue );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_s
#endif /* MAGMASPARSE_S_H */
