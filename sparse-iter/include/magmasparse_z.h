/*
 -- MAGMA (version 1.1) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date

 @precisions normal z -> s d c
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_Z_H
#define MAGMASPARSE_Z_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_z


#ifdef __cplusplus
extern "C" {
#endif


/* ////////////////////////////////////////////////////////////////////////////
 -- For backwards compatability, map old (1.6.1) to new (1.6.2) function names
*/

#define magma_z_mtranspose  magma_zmtranspose
#define magma_z_mtransfer   magma_zmtransfer
#define magma_z_vtransfer   magma_zmtransfer
#define magma_z_mconvert    magma_zmconvert
#define magma_z_vinit       magma_zvinit
#define magma_z_vvisu       magma_zprint_vector
#define magma_z_vread       magma_zvread
#define magma_z_vspread     magma_zvspread
#define magma_z_mvisu       magma_zprint_matrix
#define magma_z_mfree       magma_zmfree
#define magma_z_vfree       magma_zmfree
#define write_z_csr_mtx     magma_zwrite_csr_mtx
#define write_z_csrtomtx    magma_zwrite_csrtomtx
#define print_z_csr         magma_zprint_csr_mtx


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t
magma_zparse_opts( 
    int argc, 
    char** argv, 
    magma_zopts *opts, 
    int *matrices, 
    magma_queue_t queue );

magma_int_t 
read_z_csr_from_binary( 
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    const char * filename,
    magma_queue_t queue );

magma_int_t 
read_z_csr_from_mtx( 
    magma_storage_t *type, 
    magma_location_t *location,
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_z_csr_mtx( 
    magma_z_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zcsrset( 
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col, 
    magmaDoubleComplex *val,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_zcsrget( 
    magma_z_matrix A,
    magma_int_t *m, 
    magma_int_t *n, 
    magma_index_t **row, 
    magma_index_t **col, 
    magmaDoubleComplex **val,
    magma_queue_t queue );


magma_int_t 
magma_zvset( 
    magma_int_t m, 
    magma_int_t n, 
    magmaDoubleComplex *val,
    magma_z_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_zvget( 
    magma_z_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDoubleComplex **val,
    magma_queue_t queue );

magma_int_t 
magma_zvset_dev( 
    magma_int_t m, 
    magma_int_t n, 
    magmaDoubleComplex_ptr val,
    magma_z_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_zvget_dev( 
    magma_z_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDoubleComplex_ptr *val,
    magma_queue_t queue );


magma_int_t 
magma_z_csr_mtxsymm( 
    magma_z_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_z_csr_compressor( 
    magmaDoubleComplex ** val, 
    magma_index_t ** row, 
    magma_index_t ** col, 
    magmaDoubleComplex ** valn, 
    magma_index_t ** rown, 
    magma_index_t ** coln, 
    magma_int_t *n,
    magma_queue_t queue );

magma_int_t
magma_zmcsrcompressor( 
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zmcsrcompressor_gpu( 
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zvtranspose( 
    magma_z_matrix x,
    magma_z_matrix *y,
    magma_queue_t queue );

magma_int_t 
magma_z_cucsrtranspose( 
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
z_transpose_csr( 
    magma_int_t n_rows, 
    magma_int_t n_cols, 
    magma_int_t nnz,
    magmaDoubleComplex *val, 
    magma_index_t *row, 
    magma_index_t *col, 
    magma_int_t *new_n_rows, 
    magma_int_t *new_n_cols, 
    magma_int_t *new_nnz, 
    magmaDoubleComplex **new_val, 
    magma_index_t **new_row, 
    magma_index_t **new_col,
    magma_queue_t queue );

magma_int_t
magma_zcsrsplit( 
    magma_int_t bsize,
    magma_z_matrix A,
    magma_z_matrix *D,
    magma_z_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_zmscale( 
    magma_z_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t 
magma_zmdiff( 
    magma_z_matrix A, 
    magma_z_matrix B, 
 real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_zmdiagadd( 
    magma_z_matrix *A, 
    magmaDoubleComplex add,
    magma_queue_t queue );

magma_int_t 
magma_zmsort( 
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zindexsort(
    magma_index_t *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_zdomainoverlap(
    magma_index_t num_rows,
    magma_index_t *num_indices,
    magma_index_t *rowptr,
    magma_index_t *colidx,
    magma_index_t *x,
    magma_queue_t queue );

magma_int_t
magma_zsymbilu( 
    magma_z_matrix *A, 
    magma_int_t levels,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );


magma_int_t 
magma_zwrite_csr_mtx( 
    magma_z_matrix A,
    magma_order_t MajorType,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zwrite_csrtomtx( 
    magma_z_matrix A,
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zprint_csr( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    magma_queue_t queue );

magma_int_t 
magma_zprint_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
    magma_queue_t queue );


magma_int_t 
magma_zmtranspose(
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );


magma_int_t 
magma_zmtransfer(
    magma_z_matrix A, 
    magma_z_matrix *B, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_zmconvert(
    magma_z_matrix A, 
    magma_z_matrix *B, 
    magma_storage_t old_format, 
    magma_storage_t new_format,
    magma_queue_t queue );


magma_int_t
magma_zvinit(
    magma_z_matrix *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magma_int_t num_cols,
    magmaDoubleComplex values,
    magma_queue_t queue );

magma_int_t
magma_zprint_vector(
    magma_z_matrix x, 
    magma_int_t offset, 
    magma_int_t displaylength,
    magma_queue_t queue );

magma_int_t
magma_zvread(
    magma_z_matrix *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue );

magma_int_t
magma_zvspread(
    magma_z_matrix *x, 
    const char * filename,
    magma_queue_t queue );

magma_int_t
magma_zprint_matrix(
    magma_z_matrix A,
    magma_queue_t queue );

magma_int_t 
magma_zdiameter(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_zrowentries(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zmfree(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zresidual(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix x, 
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zresidualvec(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix x,
    magma_z_matrix *r,
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zmgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    magmaDoubleComplex *diag_vals,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zm_27stencil(
    magma_int_t n,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zm_5stencil(
    magma_int_t n,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zsolverinfo(
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zsolverinfo_init(
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zeigensolverinfo_init(
    magma_z_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_zsolverinfo_free(
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative incomplete factorizations
*/


magma_int_t
magma_ziterilusetup( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zitericsetup( 
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zitericupdate( 
    magma_z_matrix A, 
    magma_z_preconditioner *precond, 
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_zapplyiteric_l( 
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplyiteric_r( 
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_ziterilu_csr( 
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_queue_t queue );

magma_int_t
magma_ziteric_csr( 
    magma_z_matrix A,
    magma_z_matrix A_CSR,
    magma_queue_t queue );

magma_int_t 
magma_zfrobenius( 
    magma_z_matrix A, 
    magma_z_matrix B, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_znonlinres(   
    magma_z_matrix A, 
    magma_z_matrix L,
    magma_z_matrix U, 
    magma_z_matrix *LU, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_zilures(   
    magma_z_matrix A, 
    magma_z_matrix L,
    magma_z_matrix U, 
    magma_z_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_zicres(       
    magma_z_matrix A, 
    magma_z_matrix C,
    magma_z_matrix CT, 
    magma_z_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_zinitguess( 
    magma_z_matrix A, 
    magma_z_matrix *L, 
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t 
magma_zinitrecursiveLU( 
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmLdiagadd( 
    magma_z_matrix *L,
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
magma_zcg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_zcg_res(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_zcg_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_zgmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x, 
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_merge2(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpcg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbpcg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpbicgstab(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpgmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zfgmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zjacobi(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zjacobidomainoverlap(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbaiter(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ziterref(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zilu(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_zbcsrlu(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_zbcsrlutrf(
    magma_z_matrix A, 
    magma_z_matrix *M,
    magma_int_t *ipiv, 
    magma_int_t version,
    magma_queue_t queue );

magma_int_t
magma_zbcsrlusv(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );



magma_int_t
magma_zilucg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zilugmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue ); 


magma_int_t
magma_zlobpcg_shift(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magma_int_t shift,
    magmaDoubleComplex_ptr x,
    magma_queue_t queue );

magma_int_t
magma_zlobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    double *evalues, 
    magmaDoubleComplex_ptr X,
    magmaDoubleComplex_ptr R, 
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zlobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaDoubleComplex_ptr X,
    magmaDoubleComplex_ptr Y,
    magma_queue_t queue );


/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_zlobpcg(
    magma_z_matrix A, 
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par, 
    magma_queue_t queue );




/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_zjacobisetup(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *M, 
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zjacobisetup_matrix(
    magma_z_matrix A, 
    magma_z_matrix *M, 
    magma_z_matrix *d,
    magma_queue_t queue );

magma_int_t
magma_zjacobisetup_vector(
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zjacobiiter(
    magma_z_matrix M, 
    magma_z_matrix c, 
    magma_z_matrix *x, 
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zjacobiiter_precond( 
    magma_z_matrix M, 
    magma_z_matrix *x, 
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zjacobiiter_sys(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix t, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpastixsetup(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zapplypastix(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


// custom preconditioner
magma_int_t
magma_zapplycustomprecond_l(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycustomprecond_r(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


// CUSPARSE preconditioner

magma_int_t
magma_zcuilusetup(
    magma_z_matrix A, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuilu_l(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuilu_r(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zcuiccsetup(
    magma_z_matrix A, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuicc_l(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuicc_r(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zcumilusetup(
    magma_z_matrix A, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcumilugeneratesolverinfo(
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumilu_l(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumilu_r(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zcumiccsetup(
    magma_z_matrix A, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcumicgeneratesolverinfo(
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumicc_l(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumicc_r(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


// block-asynchronous iteration

magma_int_t
magma_zbajac_csr(
    magma_int_t localiters,
    magma_z_matrix D,
    magma_z_matrix R,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_z_spmv(
    magmaDoubleComplex alpha, 
    magma_z_matrix A, 
    magma_z_matrix x, 
    magmaDoubleComplex beta, 
    magma_z_matrix y,
    magma_queue_t queue );

magma_int_t
magma_zcustomspmv(
    magmaDoubleComplex alpha, 
    magma_z_matrix x, 
    magmaDoubleComplex beta, 
    magma_z_matrix y,
    magma_queue_t queue );

magma_int_t
magma_z_spmv_shift(
    magmaDoubleComplex alpha, 
    magma_z_matrix A, 
    magmaDoubleComplex lambda,
    magma_z_matrix x, 
    magmaDoubleComplex beta, 
    magma_int_t offset, 
    magma_int_t blocksize,
    magmaIndex_ptr dadd_vecs, 
    magma_z_matrix y,
    magma_queue_t queue );

magma_int_t
magma_zcuspmm(
    magma_z_matrix A, 
    magma_z_matrix B, 
    magma_z_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_z_spmm(
    magmaDoubleComplex alpha, 
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *C,
    magma_queue_t queue );

magma_int_t
magma_zsymbilu( 
    magma_z_matrix *A, 
    magma_int_t levels, 
    magma_z_matrix *L, 
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zcuspaxpy(
    magmaDoubleComplex_ptr alpha, magma_z_matrix A, 
    magmaDoubleComplex_ptr beta, magma_z_matrix B, 
    magma_z_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_z_precond(
    magma_z_matrix A, 
    magma_z_matrix b, magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_z_solver(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_zopts *zopts,
    magma_queue_t queue );

magma_int_t
magma_z_precondsetup(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_z_applyprecond(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_z_applyprecond_left(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_z_applyprecond_right(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_z_initP2P(
    magma_int_t *bandwidth_benchmark,
    magma_int_t *num_gpus,
    magma_queue_t queue );

magma_int_t
magma_zcompact(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *dnorms, double tol, 
    magma_int_t *activeMask, magma_int_t *cBlockSize,
    magma_queue_t queue );

magma_int_t
magma_zcompactActive(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magma_int_t *active,
    magma_queue_t queue );

magma_int_t
magma_zmlumerge(    
    magma_z_matrix L, 
    magma_z_matrix U,
    magma_z_matrix *A, 
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_zgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgecsrmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zmgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeellmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_zmgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_zgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    int offset,
    int blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_zmgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeellrtmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowlength,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_int_t num_threads,
    magma_int_t threads_per_row,
    magma_queue_t queue );

magma_int_t 
magma_zgesellcmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zmgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zmgesellpmv_blocked(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t
magma_zmergedgs(
    magma_int_t n, 
    magma_int_t ldh,
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zcopyscale(    
    int n, 
    int k,
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dznrm2scale(    
    int m, 
    magmaDoubleComplex_ptr dr,    
    int lddr, 
    magmaDoubleComplex *drnorm,
    magma_queue_t queue );


magma_int_t
magma_zjacobisetup_vector_gpu(
    int num_rows, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix c,
    magma_z_matrix *x,
    magma_queue_t queue );


magma_int_t
magma_zjacobi_diagscal(    
    int num_rows, 
    magma_z_matrix d, 
    magma_z_matrix b, 
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zjacobiupdate(
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobispmvupdate(
    magma_int_t maxiter,
    magma_z_matrix A, 
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobispmvupdate_bw(
    magma_int_t maxiter,
    magma_z_matrix A, 
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobispmvupdateselect(
    magma_int_t maxiter,
    magma_int_t num_updates,
    magma_index_t *indices,
    magma_z_matrix A,
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix tmp, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobisetup_diagscal(
    magma_z_matrix A, magma_z_matrix *d,
    magma_queue_t queue );


magma_int_t
magma_zbicgmerge1(    
    int n, 
    magmaDoubleComplex_ptr dskp,
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dp,
    magma_queue_t queue );


magma_int_t
magma_zbicgmerge2(
    int n, 
    magmaDoubleComplex_ptr dskp, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr ds,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge3(
    int n, 
    magmaDoubleComplex_ptr dskp, 
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr dx, 
    magmaDoubleComplex_ptr dr,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge4(
    int type, 
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zcgmerge_spmv1( 
    magma_z_matrix A,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zcgmerge_xrbeta( 
    int n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz, 
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zmdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zgemvmdot(
    int n, 
    int k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge_spmv1( 
    magma_z_matrix A,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge_spmv2( 
    magma_z_matrix A,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge_xrbeta( 
    int n,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr drr,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr dx, 
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbcsrswp(
    magma_int_t n,
    magma_int_t size_b, 
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_zbcsrtrsv(
    magma_uplo_t uplo,
    magma_int_t r_blocks,
    magma_int_t c_blocks,
    magma_int_t size_b, 
    magmaDoubleComplex_ptr dA,
    magma_index_t *blockinfo, 
    magmaDoubleComplex_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_zbcsrvalcpy(
    magma_int_t size_b, 
    magma_int_t num_blocks, 
    magma_int_t num_zero_blocks, 
    magmaDoubleComplex_ptr *dAval, 
    magmaDoubleComplex_ptr *dBval,
    magmaDoubleComplex_ptr *dBval2,
    magma_queue_t queue );

magma_int_t
magma_zbcsrluegemm(
    magma_int_t size_b, 
    magma_int_t num_block_rows,
    magma_int_t kblocks,
    magmaDoubleComplex_ptr *dA, 
    magmaDoubleComplex_ptr *dB, 
    magmaDoubleComplex_ptr *dC,
    magma_queue_t queue );

magma_int_t
magma_zbcsrlupivloc(
    magma_int_t size_b, 
    magma_int_t kblocks,
    magmaDoubleComplex_ptr *dA, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_zbcsrblockinfo5(
    magma_int_t lustep,
    magma_int_t num_blocks, 
    magma_int_t c_blocks, 
    magma_int_t size_b,
    magma_index_t *blockinfo,
    magmaDoubleComplex_ptr dval,
    magmaDoubleComplex_ptr *AII,
    magma_queue_t queue );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_Z_H */
