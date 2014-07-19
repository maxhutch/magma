/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_Z_H
#define MAGMASPARSE_Z_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Matrix Descriptors
*/
/* CSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   m;
    magma_int_t   n;
    magma_int_t nnz;

    magmaDoubleComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_zmatrix_t;


/* BCSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   rows_block;
    magma_int_t   cols_block;

    magma_int_t nrow_blocks;
    magma_int_t  nnz_blocks;

    magmaDoubleComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_zbcsr_t;


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t 
read_z_csr_from_binary( magma_int_t* n_row, 
                        magma_int_t* n_col, 
                        magma_int_t* nnz, 
                        magmaDoubleComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col,
                        const char * filename);

magma_int_t 
read_z_csr_from_mtx(    magma_storage_t *type, 
                        magma_location_t *location,
                        magma_int_t* n_row, 
                        magma_int_t* n_col, 
                        magma_int_t* nnz, 
                        magmaDoubleComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        const char *filename);

magma_int_t 
magma_z_csr_mtx(        magma_z_sparse_matrix *A, 
                        const char *filename );

magma_int_t 
magma_z_csr_mtxsymm(    magma_z_sparse_matrix *A, 
                        const char *filename );

magma_int_t 
magma_z_csr_compressor( magmaDoubleComplex ** val, 
                        magma_index_t ** row, 
                        magma_index_t ** col, 
                        magmaDoubleComplex ** valn, 
                        magma_index_t ** rown, 
                        magma_index_t ** coln, 
                        magma_int_t *n );

magma_int_t 
magma_z_csrtranspose(   magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B );

magma_int_t 
magma_z_cucsrtranspose( magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B );

magma_int_t 
z_transpose_csr(        magma_int_t n_rows, 
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
                        magma_index_t **new_col );

magma_int_t
magma_zcsrsplit(        magma_int_t bsize,
                        magma_z_sparse_matrix A,
                        magma_z_sparse_matrix *D,
                        magma_z_sparse_matrix *R );

magma_int_t
magma_zmscale(          magma_z_sparse_matrix *A, 
                        magma_scale_t scaling );

magma_int_t
magma_zmdiagadd(        magma_z_sparse_matrix *A, 
                        magmaDoubleComplex add );

magma_int_t 
magma_zmsort(           magma_z_sparse_matrix *A );

magma_int_t
magma_zmhom(            magma_z_sparse_matrix A, 
                        magma_int_t b,  
                        magma_index_t *p );

magma_int_t
magma_zmhom_fd(         magma_z_sparse_matrix A, 
                        magma_int_t n,
                        magma_int_t b,  
                        magma_index_t *p );

magma_int_t 
magma_zilustruct(       magma_z_sparse_matrix *A,
                        magma_int_t levels );

magma_int_t 
magma_z_mpkinfo_one(    magma_z_sparse_matrix A, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s,    
                        magma_index_t **num_add_rows,
                        magma_index_t **add_rows,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs );

magma_int_t 
magma_z_mpkback(        magma_z_sparse_matrix A, 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs,
                        magma_int_t *num_vecs_back,
                        magma_index_t **vecs_back );

magma_int_t 
magma_z_mpkinfo(        magma_z_sparse_matrix A, 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s,
                        magma_index_t **num_add_rows,
                        magma_index_t **add_rows,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs,
                        magma_int_t *num_vecs_back,
                        magma_index_t **vecs_back );

magma_int_t 
magma_z_mpksetup_one(   magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s );

magma_int_t 
magma_z_mpksetup(       magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix B[MagmaMaxGPUs], 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s );

magma_int_t 
magma_z_mpk_compress(   magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompress( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompress_sel( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magma_int_t offset,
                        magma_int_t blocksize,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_compress_gpu( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompress_gpu( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompspmv(  magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y );

magma_int_t
magma_z_mpk_mcompresso(  magma_z_sparse_matrix A,
                         magma_z_sparse_matrix *B,
                         magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows );

magma_int_t 
write_z_csr_mtx(        magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        magmaDoubleComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        magma_order_t MajorType,
                        const char *filename );

magma_int_t 
write_z_csrtomtx(        magma_z_sparse_matrix A,
                        const char *filename );

magma_int_t 
print_z_csr(            magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        magmaDoubleComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col );

magma_int_t 
print_z_csr_mtx(        magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        magmaDoubleComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        magma_order_t MajorType );





magma_int_t 
magma_z_mtranspose(     magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B );


magma_int_t 
magma_z_mtransfer(      magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B, 
                        magma_location_t src, 
                        magma_location_t dst );

magma_int_t 
magma_z_vtransfer(      magma_z_vector x, 
                        magma_z_vector *y, 
                        magma_location_t src, 
                        magma_location_t dst );

magma_int_t 
magma_z_mconvert(       magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B, 
                        magma_storage_t old_format, 
                        magma_storage_t new_format );

magma_int_t 
magma_z_LUmerge(        magma_z_sparse_matrix L, 
                        magma_z_sparse_matrix U, 
                        magma_z_sparse_matrix *B );

magma_int_t 
magma_z_LUmergein(      magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix L,
                        magma_z_sparse_matrix *B );

magma_int_t
magma_z_vinit(          magma_z_vector *x, 
                        magma_location_t memory_location,
                        magma_int_t num_rows, 
                        magmaDoubleComplex values );

magma_int_t
magma_z_vvisu(          magma_z_vector x, 
                        magma_int_t offset, 
                        magma_int_t displaylength );

magma_int_t
magma_z_vread(          magma_z_vector *x, 
                        magma_int_t length,
                        char * filename );
magma_int_t
magma_z_mvisu(          magma_z_sparse_matrix A );

magma_int_t 
magma_zdiameter(        magma_z_sparse_matrix *A );

magma_int_t 
magma_zrowentries(      magma_z_sparse_matrix *A );

magma_int_t
magma_z_mfree(          magma_z_sparse_matrix *A );

magma_int_t
magma_z_vfree(          magma_z_vector *x );

magma_int_t
magma_zresidual(        magma_z_sparse_matrix A, 
                        magma_z_vector b, 
                        magma_z_vector x, 
                        double *res );

magma_int_t
magma_zbitflip(     magmaDoubleComplex *d, 
                    magma_int_t loc, 
                    magma_int_t bit );

magma_int_t
magma_zmgenerator(  magma_int_t n,
                    magma_int_t offdiags,
                    magma_index_t *diag_offset,
                    magmaDoubleComplex *diag_vals,
                    magma_z_sparse_matrix *A );

magma_int_t
magma_zsolverinfo(  magma_z_solver_par *solver_par, 
                    magma_z_preconditioner *precond_par );

magma_int_t
magma_zsolverinfo_init( magma_z_solver_par *solver_par, 
                        magma_z_preconditioner *precond );

magma_int_t
magma_zsolverinfo_free( magma_z_solver_par *solver_par, 
                        magma_z_preconditioner *precond );


magma_int_t 
magma_zfrobenius( magma_z_sparse_matrix A, magma_z_sparse_matrix B, 
                  real_Double_t *res );

magma_int_t 
magma_znonlinres(   magma_z_sparse_matrix A, 
                    magma_z_sparse_matrix L,
                    magma_z_sparse_matrix U, 
                    magma_z_sparse_matrix *LU, 
                    real_Double_t *res );

magma_int_t 
magma_zilures(      magma_z_sparse_matrix A, 
                    magma_z_sparse_matrix L,
                    magma_z_sparse_matrix U, 
                    magma_z_sparse_matrix *LU, 
                    real_Double_t *res );

magma_int_t 
magma_zinitguess(   magma_z_sparse_matrix A, 
                    magma_z_sparse_matrix *L,
                    magma_z_sparse_matrix *U );

magma_int_t
magma_zmreorder( magma_z_sparse_matrix A, magma_int_t n, magma_int_t b, magma_z_sparse_matrix *B );

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
magma_zcg(             magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t 
magma_zcg_res(         magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t 
magma_zcg_sdc(         magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t 
magma_zcg_merge(       magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zcg_exactres(    magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t 
magma_zgmres(          magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zbicgstab(       magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_z_solver_par *solver_par );

magma_int_t
magma_zbicgstab_merge( magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zbicgstab_merge2( magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zpcg(            magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par, 
                       magma_z_preconditioner *precond_par );

magma_int_t
magma_zpbicgstab(      magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par, 
                       magma_z_preconditioner *precond_par );

magma_int_t
magma_zpgmres(         magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par, 
                       magma_z_preconditioner *precond_par );
magma_int_t
magma_zjacobi(         magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zbaiter(         magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_ziterref(        magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par, 
                       magma_z_preconditioner *precond_par );
magma_int_t
magma_zp1gmres(        magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zgmres_pipe(     magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zilu(            magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par, 
                       magma_int_t *ipiv );

magma_int_t
magma_zbcsrlu(         magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );


magma_int_t
magma_zbcsrlutrf(      magma_z_sparse_matrix A, 
                       magma_z_sparse_matrix *M,
                       magma_int_t *ipiv, 
                       magma_int_t version );

magma_int_t
magma_zbcsrlusv(       magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par, 
                       magma_int_t *ipiv );



magma_int_t
magma_zilucg(          magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par );

magma_int_t
magma_zilugmres(       magma_z_sparse_matrix A, magma_z_vector b, 
                       magma_z_vector *x, magma_z_solver_par *solver_par ); 


magma_int_t
magma_zlobpcg_shift(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magma_int_t shift,
                        magmaDoubleComplex *x );
magma_int_t
magma_zlobpcg_res(      magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        double *evalues, 
                        magmaDoubleComplex *X,
                        magmaDoubleComplex *R, 
                        double *res );
magma_int_t
magma_zlobpcg_maxpy(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magmaDoubleComplex *X,
                        magmaDoubleComplex *Y);

/*/////////////////////////////////////////////////////////////////////////////
    -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_zlobpcg(          magma_z_sparse_matrix A,
                        magma_z_solver_par *solver_par );




/*/////////////////////////////////////////////////////////////////////////////
    -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_zjacobisetup(     magma_z_sparse_matrix A, 
                        magma_z_vector b, 
                        magma_z_sparse_matrix *M, 
                        magma_z_vector *c );
magma_int_t
magma_zjacobisetup_matrix(  magma_z_sparse_matrix A, 
                            magma_z_vector b, 
                            magma_z_sparse_matrix *M, 
                            magma_z_vector *d );
magma_int_t
magma_zjacobisetup_vector(  magma_z_vector b,  
                            magma_z_vector d, 
                            magma_z_vector *c );

magma_int_t
magma_zjacobiiter(      magma_z_sparse_matrix M, 
                        magma_z_vector c, 
                        magma_z_vector *x,  
                        magma_z_solver_par *solver_par );

magma_int_t
magma_zilusetup(        magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *M,
                        magma_int_t *ipiv );

magma_int_t
magma_zpastixsetup(     magma_z_sparse_matrix A, magma_z_vector b,
                        magma_z_preconditioner *precond );


magma_int_t
magma_zapplypastix(     magma_z_vector b, magma_z_vector *x, 
                        magma_z_preconditioner *precond );


// CUSPARSE preconditioner

magma_int_t
magma_zcuilusetup( magma_z_sparse_matrix A, magma_z_preconditioner *precond );

magma_int_t
magma_zapplycuilu_l( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond );
magma_int_t
magma_zapplycuilu_r( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond );


magma_int_t
magma_zcuiccsetup( magma_z_sparse_matrix A, magma_z_preconditioner *precond );

magma_int_t
magma_zapplycuicc_l( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond );
magma_int_t
magma_zapplycuicc_r( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond );

// block-asynchronous preconditioner

magma_int_t
magma_zailusetup( magma_z_sparse_matrix A, magma_z_preconditioner *precond );

magma_int_t
magma_zaiccsetup( magma_z_sparse_matrix A, magma_z_preconditioner *precond );

magma_int_t
magma_zapplyailu_l( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond );
magma_int_t
magma_zapplyailu_r( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond );


// block-asynchronous iteration

magma_int_t
magma_zbajac_csr(   magma_int_t localiters,
                    magma_z_sparse_matrix D,
                    magma_z_sparse_matrix R,
                    magma_z_vector b,
                    magma_z_vector *x );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_z_spmv(           magmaDoubleComplex alpha, 
                        magma_z_sparse_matrix A, 
                        magma_z_vector x, 
                        magmaDoubleComplex beta, 
                        magma_z_vector y );

magma_int_t
magma_z_spmv_shift(     magmaDoubleComplex alpha, 
                        magma_z_sparse_matrix A, 
                        magmaDoubleComplex lambda,
                        magma_z_vector x, 
                        magmaDoubleComplex beta, 
                        magma_int_t offset,     
                        magma_int_t blocksize,
                        magma_index_t *add_vecs, 
                        magma_z_vector y );

magma_int_t
magma_zcuspmm(          magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix B, 
                        magma_z_sparse_matrix *AB );

magma_int_t
magma_z_precond(        magma_z_sparse_matrix A, 
                        magma_z_vector b, magma_z_vector *x,
                        magma_z_preconditioner precond );

magma_int_t
magma_z_precondsetup( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_preconditioner *precond );

magma_int_t
magma_z_applyprecond( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond );


magma_int_t
magma_z_applyprecond_left( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond );


magma_int_t
magma_z_applyprecond_right( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond );

magma_int_t
magma_zorthomgs(        magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magmaDoubleComplex *X );

magma_int_t
magma_z_initP2P(        magma_int_t *bandwidth_benchmark,
                        magma_int_t *num_gpus );




/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_zgecsrmv(        magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zgecsrmv_shift(  magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex lambda,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zmgecsrmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zgeellmv(        magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zgeellmv_shift(  magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex lambda,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaDoubleComplex *d_y );


magma_int_t 
magma_zmgeellmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );


magma_int_t 
magma_zgeelltmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zgeelltmv_shift( magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex lambda,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaDoubleComplex *d_y );


magma_int_t 
magma_zmgeelltmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zgeellrtmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magma_index_t *d_rowlength,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y,
                       magma_int_t num_threads,
                       magma_int_t threads_per_row );

magma_int_t 
magma_zgesellcmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t blocksize,
                       magma_int_t slices,
                       magma_int_t alignment,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_index_t *d_colind,
                       magma_index_t *d_rowptr,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t
magma_zgesellpmv(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y );

magma_int_t
magma_zmgesellpmv( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y );

magma_int_t
magma_zmgesellpmv_blocked( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y );


magma_int_t 
magma_zp1gmres_mgs(    magma_int_t  n, 
                       magma_int_t  k, 
                       magmaDoubleComplex *skp, 
                       magmaDoubleComplex *v, 
                       magmaDoubleComplex *z );


magma_int_t
magma_zmergedgs(        magma_int_t n, 
                        magma_int_t ldh,
                        magma_int_t k, 
                        magmaDoubleComplex *v, 
                        magmaDoubleComplex *r,
                        magmaDoubleComplex *skp );

magma_int_t
magma_zcopyscale(       int n, 
                        int k,
                        magmaDoubleComplex *r, 
                        magmaDoubleComplex *v,
                        magmaDoubleComplex *skp );

magma_int_t
magma_dznrm2scale(      int m, 
                        magmaDoubleComplex *r, int lddr, 
                        magmaDoubleComplex *drnorm);


magma_int_t
magma_zjacobisetup_vector_gpu( int num_rows, 
                               magmaDoubleComplex *b, 
                               magmaDoubleComplex *d, 
                               magmaDoubleComplex *c);

magma_int_t
magma_zjacobi_diagscal(         int num_rows, 
                                magmaDoubleComplex *b, 
                                magmaDoubleComplex *d, 
                                magmaDoubleComplex *c);

magma_int_t
magma_zjacobisetup_diagscal( magma_z_sparse_matrix A, magma_z_vector *d );


magma_int_t
magma_zbicgmerge1(  int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *p );


magma_int_t
magma_zbicgmerge2(  int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *s );

magma_int_t
magma_zbicgmerge3(  int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *s,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r );
magma_int_t
magma_zbicgmerge4(  int type, 
                    magmaDoubleComplex *skp );

magma_int_t
magma_zcgmerge_spmv1(  
                 magma_z_sparse_matrix A,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_d,
                 magmaDoubleComplex *d_z,
                 magmaDoubleComplex *skp );

magma_int_t
magma_zcgmerge_xrbeta(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *x,
                 magmaDoubleComplex *r,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z, 
                 magmaDoubleComplex *skp );

magma_int_t
magma_zmdotc(       magma_int_t n, 
                    magma_int_t k, 
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d1,
                    magmaDoubleComplex *d2,
                    magmaDoubleComplex *skp );

magma_int_t
magma_zgemvmdot(    int n, 
                    int k, 
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d1,
                    magmaDoubleComplex *d2,
                    magmaDoubleComplex *skp );

magma_int_t
magma_zbicgmerge_spmv1(  
                 magma_z_sparse_matrix A,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_p,
                 magmaDoubleComplex *d_r,
                 magmaDoubleComplex *d_v,
                 magmaDoubleComplex *skp );

magma_int_t
magma_zbicgmerge_spmv2(  
                 magma_z_sparse_matrix A,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_s,
                 magmaDoubleComplex *d_t,
                 magmaDoubleComplex *skp );

magma_int_t
magma_zbicgmerge_xrbeta(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *rr,
                 magmaDoubleComplex *r,
                 magmaDoubleComplex *p,
                 magmaDoubleComplex *s,
                 magmaDoubleComplex *t,
                 magmaDoubleComplex *x, 
                 magmaDoubleComplex *skp );

magma_int_t
magma_zbcsrswp(  magma_int_t n,
                 magma_int_t size_b, 
                 magma_int_t *ipiv,
                 magmaDoubleComplex *x );

magma_int_t
magma_zbcsrtrsv( magma_uplo_t uplo,
                 magma_int_t r_blocks,
                 magma_int_t c_blocks,
                 magma_int_t size_b, 
                 magmaDoubleComplex *A,
                 magma_index_t *blockinfo,   
                 magmaDoubleComplex *x );

magma_int_t
magma_zbcsrvalcpy(  magma_int_t size_b, 
                    magma_int_t num_blocks, 
                    magma_int_t num_zero_blocks, 
                    magmaDoubleComplex **Aval, 
                    magmaDoubleComplex **Bval,
                    magmaDoubleComplex **Bval2 );

magma_int_t
magma_zbcsrluegemm( magma_int_t size_b, 
                    magma_int_t num_block_rows,
                    magma_int_t kblocks,
                    magmaDoubleComplex **dA,  
                    magmaDoubleComplex **dB,  
                    magmaDoubleComplex **dC );

magma_int_t
magma_zbcsrlupivloc( magma_int_t size_b, 
                    magma_int_t kblocks,
                    magmaDoubleComplex **dA,  
                    magma_int_t *ipiv );

magma_int_t
magma_zbcsrblockinfo5(  magma_int_t lustep,
                        magma_int_t num_blocks, 
                        magma_int_t c_blocks, 
                        magma_int_t size_b,
                        magma_index_t *blockinfo,
                        magmaDoubleComplex *val,
                        magmaDoubleComplex **AII );

magma_int_t
magma_zgesellpmv_mpk(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y );



magma_int_t
magma_zilu_c( magma_z_sparse_matrix A,
                    magma_z_sparse_matrix A_ELLDD );

magma_int_t
magma_zilu_s( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_ELLDD );

magma_int_t
magma_zilu_cs(    magma_z_sparse_matrix A,
                    magma_z_sparse_matrix A_ELLDD );

magma_int_t
magma_zilu_ss( magma_z_sparse_matrix A,
                    magma_z_sparse_matrix A_ELLDD );
magma_int_t
magma_zilu_st( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_ELLDD );

magma_int_t
magma_zicc_ss( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_ELLD );

magma_int_t
magma_zicc_cs( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_ELLDD );

magma_int_t
magma_zilu_csr( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSRCSC );

magma_int_t
magma_zaic_csr_a( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSR );

magma_int_t
magma_zaic_csr_s( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSR );

magma_int_t
magma_zaic_csr_cs( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSR );


magma_int_t
magma_zaic_csr_ss( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSR );


magma_int_t
magma_zailu_csr_s( magma_z_sparse_matrix A_L,
                   magma_z_sparse_matrix A_U,
                   magma_z_sparse_matrix L,
                   magma_z_sparse_matrix U );

magma_int_t
magma_zailu_csr_a( magma_z_sparse_matrix A,
                   magma_z_sparse_matrix L,
                   magma_z_sparse_matrix U );

magma_int_t
magma_zailu_csr_s_debug( magma_index_t *p,
                   magma_z_sparse_matrix A_L,
                   magma_z_sparse_matrix A_U,
                   magma_z_sparse_matrix L,
                   magma_z_sparse_matrix U );
magma_int_t
magma_zailu_csr_s_gs( magma_z_sparse_matrix A_L,
                   magma_z_sparse_matrix A_U,
                   magma_z_sparse_matrix L,
                   magma_z_sparse_matrix U,
                   magmaDoubleComplex omega );

magma_int_t
magma_zailu_csr_s_gslu( magma_z_sparse_matrix A_L,
                   magma_z_sparse_matrix A_U,
                   magma_z_sparse_matrix L,
                   magma_z_sparse_matrix U,
                   magmaDoubleComplex omega );


magma_int_t
magma_ztrisv_l( magma_z_sparse_matrix A,
                magma_z_vector b,
                magma_z_vector *x );

magma_int_t
magma_ztrisv_r( magma_z_sparse_matrix A,
                magma_z_vector b,
                magma_z_vector *x );

magma_int_t
magma_ztrisv_l_nu( magma_z_sparse_matrix A,
                magma_z_vector b,
                magma_z_vector *x );

magma_int_t
magma_ztrisv_r_nu( magma_z_sparse_matrix A,
                magma_z_vector b,
                magma_z_vector *x );

magma_int_t 
mtools_zsymmetrize( magma_z_sparse_matrix *A );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_Z_H */
