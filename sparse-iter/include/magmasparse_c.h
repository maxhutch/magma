/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magmasparse_z.h normal z -> c, Fri May 30 10:41:30 2014
       @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_C_H
#define MAGMASPARSE_C_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_c


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Matrix Descriptors
*/
/* CSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   m;
    magma_int_t   n;
    magma_int_t nnz;

    magmaFloatComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_cmatrix_t;


/* BCSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   rows_block;
    magma_int_t   cols_block;

    magma_int_t nrow_blocks;
    magma_int_t  nnz_blocks;

    magmaFloatComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_cbcsr_t;


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t 
read_c_csr_from_binary( magma_int_t* n_row, 
                        magma_int_t* n_col, 
                        magma_int_t* nnz, 
                        magmaFloatComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col,
                        const char * filename);

magma_int_t 
read_c_csr_from_mtx(    magma_storage_t *type, 
                        magma_location_t *location,
                        magma_int_t* n_row, 
                        magma_int_t* n_col, 
                        magma_int_t* nnz, 
                        magmaFloatComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        const char *filename);

magma_int_t 
magma_c_csr_mtx(        magma_c_sparse_matrix *A, 
                        const char *filename );

magma_int_t 
magma_c_csr_mtxsymm(    magma_c_sparse_matrix *A, 
                        const char *filename );

magma_int_t 
magma_c_csr_compressor( magmaFloatComplex ** val, 
                        magma_index_t ** row, 
                        magma_index_t ** col, 
                        magmaFloatComplex ** valn, 
                        magma_index_t ** rown, 
                        magma_index_t ** coln, 
                        magma_int_t *n,
                        magma_int_t *alignment);

magma_int_t 
magma_c_csrtranspose(   magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *B );

magma_int_t 
magma_c_cucsrtranspose( magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *B );

magma_int_t 
c_transpose_csr(        magma_int_t n_rows, 
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
                        magma_index_t **new_col );

magma_int_t
magma_ccsrsplit(        magma_int_t bsize,
                        magma_c_sparse_matrix A,
                        magma_c_sparse_matrix *D,
                        magma_c_sparse_matrix *R );

magma_int_t
magma_cmscale(          magma_c_sparse_matrix *A, 
                        magma_scale_t scaling );

magma_int_t 
magma_cmsort(           magma_c_sparse_matrix *A );

magma_int_t 
magma_cilustruct(       magma_c_sparse_matrix *A,
                        magma_int_t levels );

magma_int_t 
magma_c_mpkinfo_one(    magma_c_sparse_matrix A, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s,    
                        magma_index_t **num_add_rows,
                        magma_index_t **add_rows,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs );

magma_int_t 
magma_c_mpkback(        magma_c_sparse_matrix A, 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs,
                        magma_int_t *num_vecs_back,
                        magma_index_t **vecs_back );

magma_int_t 
magma_c_mpkinfo(        magma_c_sparse_matrix A, 
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
magma_c_mpksetup_one(   magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *B, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s );

magma_int_t 
magma_c_mpksetup(       magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix B[MagmaMaxGPUs], 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s );

magma_int_t 
magma_c_mpk_compress(   magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaFloatComplex *x,
                        magmaFloatComplex *y );

magma_int_t 
magma_c_mpk_uncompress( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaFloatComplex *x,
                        magmaFloatComplex *y );

magma_int_t 
magma_c_mpk_uncompress_sel( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magma_int_t offset,
                        magma_int_t blocksize,
                        magmaFloatComplex *x,
                        magmaFloatComplex *y );

magma_int_t 
magma_c_mpk_compress_gpu( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaFloatComplex *x,
                        magmaFloatComplex *y );

magma_int_t 
magma_c_mpk_uncompress_gpu( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magmaFloatComplex *x,
                        magmaFloatComplex *y );

magma_int_t 
magma_c_mpk_uncompspmv(  magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows,
                         magmaFloatComplex *x,
                         magmaFloatComplex *y );

magma_int_t
magma_c_mpk_mcompresso(  magma_c_sparse_matrix A,
                         magma_c_sparse_matrix *B,
                         magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows );

magma_int_t 
write_c_csr_mtx(        magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        magmaFloatComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        magma_order_t MajorType,
                        const char *filename );

magma_int_t 
write_c_csrtomtx(        magma_c_sparse_matrix A,
                        const char *filename );

magma_int_t 
print_c_csr(            magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        magmaFloatComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col );

magma_int_t 
print_c_csr_mtx(        magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        magmaFloatComplex **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        magma_order_t MajorType );





magma_int_t 
magma_c_mtranspose(     magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *B );


magma_int_t 
magma_c_mtransfer(      magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *B, 
                        magma_location_t src, 
                        magma_location_t dst );

magma_int_t 
magma_c_vtransfer(      magma_c_vector x, 
                        magma_c_vector *y, 
                        magma_location_t src, 
                        magma_location_t dst );

magma_int_t 
magma_c_mconvert(       magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *B, 
                        magma_storage_t old_format, 
                        magma_storage_t new_format );

magma_int_t 
magma_c_LUmerge(        magma_c_sparse_matrix L, 
                        magma_c_sparse_matrix U, 
                        magma_c_sparse_matrix *B );

magma_int_t 
magma_c_LUmergein(      magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix L,
                        magma_c_sparse_matrix *B );

magma_int_t
magma_c_vinit(          magma_c_vector *x, 
                        magma_location_t memory_location,
                        magma_int_t num_rows, 
                        magmaFloatComplex values );

magma_int_t
magma_c_vvisu(          magma_c_vector x, 
                        magma_int_t offset, 
                        magma_int_t displaylength );

magma_int_t
magma_c_vread(          magma_c_vector *x, 
                        magma_int_t length,
                        char * filename );
magma_int_t
magma_c_mvisu(          magma_c_sparse_matrix A );

magma_int_t 
magma_cdiameter(        magma_c_sparse_matrix *A );

magma_int_t 
magma_crowentries(      magma_c_sparse_matrix *A );

magma_int_t
magma_c_mfree(          magma_c_sparse_matrix *A );

magma_int_t
magma_c_vfree(          magma_c_vector *x );

magma_int_t
magma_cresidual(        magma_c_sparse_matrix A, 
                        magma_c_vector b, 
                        magma_c_vector x, 
                        float *res );

magma_int_t
magma_cmgenerator(  magma_int_t n,
                    magma_int_t offdiags,
                    magma_index_t *diag_offset,
                    magmaFloatComplex *diag_vals,
                    magma_c_sparse_matrix *A );

magma_int_t
magma_csolverinfo(  magma_c_solver_par *solver_par, 
                    magma_c_preconditioner *precond_par );

magma_int_t
magma_csolverinfo_init( magma_c_solver_par *solver_par, 
                        magma_c_preconditioner *precond );

magma_int_t
magma_csolverinfo_free( magma_c_solver_par *solver_par, 
                        magma_c_preconditioner *precond );


magma_int_t 
magma_cfrobenius( magma_c_sparse_matrix A, magma_c_sparse_matrix B, 
                  real_Double_t *res );

magma_int_t 
magma_cnonlinres(   magma_c_sparse_matrix A, 
                    magma_c_sparse_matrix L,
                    magma_c_sparse_matrix U, 
                    magma_c_sparse_matrix *LU, 
                    real_Double_t *res );

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
magma_ccg(             magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t 
magma_ccg_merge(       magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t 
magma_cgmres(          magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cbicgstab(       magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector *x,  
                       magma_c_solver_par *solver_par );

magma_int_t
magma_cbicgstab_merge( magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cbicgstab_merge2( magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cpcg(            magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par, 
                       magma_c_preconditioner *precond_par );

magma_int_t
magma_cpbicgstab(      magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par, 
                       magma_c_preconditioner *precond_par );

magma_int_t
magma_cpgmres(         magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par, 
                       magma_c_preconditioner *precond_par );
magma_int_t
magma_cjacobi(         magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cbaiter(         magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_citerref(        magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par, 
                       magma_c_preconditioner *precond_par );
magma_int_t
magma_cp1gmres(        magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cgmres_pipe(     magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cilu(            magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par, 
                       magma_int_t *ipiv );

magma_int_t
magma_cbcsrlu(         magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );


magma_int_t
magma_cbcsrlutrf(      magma_c_sparse_matrix A, 
                       magma_c_sparse_matrix *M,
                       magma_int_t *ipiv, 
                       magma_int_t version );

magma_int_t
magma_cbcsrlusv(       magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par, 
                       magma_int_t *ipiv );



magma_int_t
magma_cilucg(          magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par );

magma_int_t
magma_cilugmres(       magma_c_sparse_matrix A, magma_c_vector b, 
                       magma_c_vector *x, magma_c_solver_par *solver_par ); 


magma_int_t
magma_clobpcg_shift(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magma_int_t shift,
                        magmaFloatComplex *x );
magma_int_t
magma_clobpcg_res(      magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        float *evalues, 
                        magmaFloatComplex *X,
                        magmaFloatComplex *R, 
                        float *res );
magma_int_t
magma_clobpcg_maxpy(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magmaFloatComplex *X,
                        magmaFloatComplex *Y);

/*/////////////////////////////////////////////////////////////////////////////
    -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_clobpcg(          magma_c_sparse_matrix A,
                        magma_c_solver_par *solver_par );




/*/////////////////////////////////////////////////////////////////////////////
    -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_cjacobisetup(     magma_c_sparse_matrix A, 
                        magma_c_vector b, 
                        magma_c_sparse_matrix *M, 
                        magma_c_vector *c );
magma_int_t
magma_cjacobisetup_matrix(  magma_c_sparse_matrix A, 
                            magma_c_vector b, 
                            magma_c_sparse_matrix *M, 
                            magma_c_vector *d );
magma_int_t
magma_cjacobisetup_vector(  magma_c_vector b,  
                            magma_c_vector d, 
                            magma_c_vector *c );

magma_int_t
magma_cjacobiiter(      magma_c_sparse_matrix M, 
                        magma_c_vector c, 
                        magma_c_vector *x,  
                        magma_c_solver_par *solver_par );

magma_int_t
magma_cilusetup(        magma_c_sparse_matrix A, 
                        magma_c_sparse_matrix *M,
                        magma_int_t *ipiv );

magma_int_t
magma_cpastixsetup(     magma_c_sparse_matrix A, magma_c_vector b,
                        magma_c_preconditioner *precond );


magma_int_t
magma_capplypastix(     magma_c_vector b, magma_c_vector *x, 
                        magma_c_preconditioner *precond );


// CUSPARSE preconditioner

magma_int_t
magma_ccuilusetup( magma_c_sparse_matrix A, magma_c_preconditioner *precond );

magma_int_t
magma_capplycuilu_l( magma_c_vector b, magma_c_vector *x, 
                    magma_c_preconditioner *precond );
magma_int_t
magma_capplycuilu_r( magma_c_vector b, magma_c_vector *x, 
                    magma_c_preconditioner *precond );


magma_int_t
magma_ccuiccsetup( magma_c_sparse_matrix A, magma_c_preconditioner *precond );

magma_int_t
magma_capplycuicc_l( magma_c_vector b, magma_c_vector *x, 
                    magma_c_preconditioner *precond );
magma_int_t
magma_capplycuicc_r( magma_c_vector b, magma_c_vector *x, 
                    magma_c_preconditioner *precond );

// block-asynchronous preconditioner

magma_int_t
magma_cailusetup( magma_c_sparse_matrix A, magma_c_preconditioner *precond );

magma_int_t
magma_caiccsetup( magma_c_sparse_matrix A, magma_c_preconditioner *precond );

magma_int_t
magma_capplyailu_l( magma_c_vector b, magma_c_vector *x, 
                    magma_c_preconditioner *precond );
magma_int_t
magma_capplyailu_r( magma_c_vector b, magma_c_vector *x, 
                    magma_c_preconditioner *precond );


// block-asynchronous iteration

magma_int_t
magma_cbajac_csr(   magma_int_t localiters,
                    magma_c_sparse_matrix D,
                    magma_c_sparse_matrix R,
                    magma_c_vector b,
                    magma_c_vector *x );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_c_spmv(           magmaFloatComplex alpha, 
                        magma_c_sparse_matrix A, 
                        magma_c_vector x, 
                        magmaFloatComplex beta, 
                        magma_c_vector y );

magma_int_t
magma_c_spmv_shift(     magmaFloatComplex alpha, 
                        magma_c_sparse_matrix A, 
                        magmaFloatComplex lambda,
                        magma_c_vector x, 
                        magmaFloatComplex beta, 
                        magma_int_t offset,     
                        magma_int_t blocksize,
                        magma_index_t *add_vecs, 
                        magma_c_vector y );

magma_int_t
magma_c_precond(        magma_c_sparse_matrix A, 
                        magma_c_vector b, magma_c_vector *x,
                        magma_c_preconditioner precond );

magma_int_t
magma_c_precondsetup( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_preconditioner *precond );

magma_int_t
magma_c_applyprecond( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_vector *x, magma_c_preconditioner *precond );


magma_int_t
magma_c_applyprecond_left( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_vector *x, magma_c_preconditioner *precond );


magma_int_t
magma_c_applyprecond_right( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_vector *x, magma_c_preconditioner *precond );

magma_int_t
magma_corthomgs(        magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magmaFloatComplex *X );

magma_int_t
magma_c_initP2P(        magma_int_t *bandwidth_benchmark,
                        magma_int_t *num_gpus );



/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_cgecsrmv(        magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );

magma_int_t 
magma_cgecsrmv_shift(  magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magmaFloatComplex alpha,
                       magmaFloatComplex lambda,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaFloatComplex *d_y );

magma_int_t 
magma_cmgecsrmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );

magma_int_t 
magma_cgeellmv(        magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );

magma_int_t 
magma_cgeellmv_shift(  magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex lambda,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaFloatComplex *d_y );


magma_int_t 
magma_cmgeellmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );


magma_int_t 
magma_cgeelltmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );

magma_int_t 
magma_cgeelltmv_shift( magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex lambda,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       magmaFloatComplex *d_y );


magma_int_t 
magma_cmgeelltmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );

magma_int_t 
magma_cgeellrtmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magma_index_t *d_rowlength,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y,
                       magma_int_t num_threads,
                       magma_int_t threads_per_row );

magma_int_t 
magma_cgesellcmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t blocksize,
                       magma_int_t slices,
                       magma_int_t alignment,
                       magmaFloatComplex alpha,
                       magmaFloatComplex *d_val,
                       magma_index_t *d_colind,
                       magma_index_t *d_rowptr,
                       magmaFloatComplex *d_x,
                       magmaFloatComplex beta,
                       magmaFloatComplex *d_y );

magma_int_t
magma_cgesellpmv(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y );

magma_int_t
magma_cmgesellpmv( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y );

magma_int_t
magma_cmgesellpmv_blocked( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y );


magma_int_t 
magma_cp1gmres_mgs(    magma_int_t  n, 
                       magma_int_t  k, 
                       magmaFloatComplex *skp, 
                       magmaFloatComplex *v, 
                       magmaFloatComplex *z );


magma_int_t
magma_cmergedgs(        magma_int_t n, 
                        magma_int_t ldh,
                        magma_int_t k, 
                        magmaFloatComplex *v, 
                        magmaFloatComplex *r,
                        magmaFloatComplex *skp );

magma_int_t
magma_ccopyscale(       int n, 
                        int k,
                        magmaFloatComplex *r, 
                        magmaFloatComplex *v,
                        magmaFloatComplex *skp );

magma_int_t
magma_scnrm2scale(      int m, 
                        magmaFloatComplex *r, int lddr, 
                        magmaFloatComplex *drnorm);


magma_int_t
magma_cjacobisetup_vector_gpu( int num_rows, 
                               magmaFloatComplex *b, 
                               magmaFloatComplex *d, 
                               magmaFloatComplex *c);

magma_int_t
magma_cjacobi_diagscal(         int num_rows, 
                                magmaFloatComplex *b, 
                                magmaFloatComplex *d, 
                                magmaFloatComplex *c);

magma_int_t
magma_cjacobisetup_diagscal( magma_c_sparse_matrix A, magma_c_vector *d );


magma_int_t
magma_cbicgmerge1(  int n, 
                    magmaFloatComplex *skp,
                    magmaFloatComplex *v, 
                    magmaFloatComplex *r, 
                    magmaFloatComplex *p );


magma_int_t
magma_cbicgmerge2(  int n, 
                    magmaFloatComplex *skp, 
                    magmaFloatComplex *r,
                    magmaFloatComplex *v, 
                    magmaFloatComplex *s );

magma_int_t
magma_cbicgmerge3(  int n, 
                    magmaFloatComplex *skp, 
                    magmaFloatComplex *p,
                    magmaFloatComplex *s,
                    magmaFloatComplex *t,
                    magmaFloatComplex *x, 
                    magmaFloatComplex *r );
magma_int_t
magma_cbicgmerge4(  int type, 
                    magmaFloatComplex *skp );

magma_int_t
magma_ccgmerge_spmv1(  
                 magma_c_sparse_matrix A,
                 magmaFloatComplex *d1,
                 magmaFloatComplex *d2,
                 magmaFloatComplex *d_d,
                 magmaFloatComplex *d_z,
                 magmaFloatComplex *skp );

magma_int_t
magma_ccgmerge_xrbeta(  
                 int n,
                 magmaFloatComplex *d1,
                 magmaFloatComplex *d2,
                 magmaFloatComplex *x,
                 magmaFloatComplex *r,
                 magmaFloatComplex *d,
                 magmaFloatComplex *z, 
                 magmaFloatComplex *skp );

magma_int_t
magma_cmdotc(       magma_int_t n, 
                    magma_int_t k, 
                    magmaFloatComplex *v, 
                    magmaFloatComplex *r,
                    magmaFloatComplex *d1,
                    magmaFloatComplex *d2,
                    magmaFloatComplex *skp );

magma_int_t
magma_cgemvmdot(    int n, 
                    int k, 
                    magmaFloatComplex *v, 
                    magmaFloatComplex *r,
                    magmaFloatComplex *d1,
                    magmaFloatComplex *d2,
                    magmaFloatComplex *skp );

magma_int_t
magma_cbicgmerge_spmv1(  
                 magma_c_sparse_matrix A,
                 magmaFloatComplex *d1,
                 magmaFloatComplex *d2,
                 magmaFloatComplex *d_p,
                 magmaFloatComplex *d_r,
                 magmaFloatComplex *d_v,
                 magmaFloatComplex *skp );

magma_int_t
magma_cbicgmerge_spmv2(  
                 magma_c_sparse_matrix A,
                 magmaFloatComplex *d1,
                 magmaFloatComplex *d2,
                 magmaFloatComplex *d_s,
                 magmaFloatComplex *d_t,
                 magmaFloatComplex *skp );

magma_int_t
magma_cbicgmerge_xrbeta(  
                 int n,
                 magmaFloatComplex *d1,
                 magmaFloatComplex *d2,
                 magmaFloatComplex *rr,
                 magmaFloatComplex *r,
                 magmaFloatComplex *p,
                 magmaFloatComplex *s,
                 magmaFloatComplex *t,
                 magmaFloatComplex *x, 
                 magmaFloatComplex *skp );

magma_int_t
magma_cbcsrswp(  magma_int_t n,
                 magma_int_t size_b, 
                 magma_int_t *ipiv,
                 magmaFloatComplex *x );

magma_int_t
magma_cbcsrtrsv( magma_uplo_t uplo,
                 magma_int_t r_blocks,
                 magma_int_t c_blocks,
                 magma_int_t size_b, 
                 magmaFloatComplex *A,
                 magma_index_t *blockinfo,   
                 magmaFloatComplex *x );

magma_int_t
magma_cbcsrvalcpy(  magma_int_t size_b, 
                    magma_int_t num_blocks, 
                    magma_int_t num_zero_blocks, 
                    magmaFloatComplex **Aval, 
                    magmaFloatComplex **Bval,
                    magmaFloatComplex **Bval2 );

magma_int_t
magma_cbcsrluegemm( magma_int_t size_b, 
                    magma_int_t num_block_rows,
                    magma_int_t kblocks,
                    magmaFloatComplex **dA,  
                    magmaFloatComplex **dB,  
                    magmaFloatComplex **dC );

magma_int_t
magma_cbcsrlupivloc( magma_int_t size_b, 
                    magma_int_t kblocks,
                    magmaFloatComplex **dA,  
                    magma_int_t *ipiv );

magma_int_t
magma_cbcsrblockinfo5(  magma_int_t lustep,
                        magma_int_t num_blocks, 
                        magma_int_t c_blocks, 
                        magma_int_t size_b,
                        magma_index_t *blockinfo,
                        magmaFloatComplex *val,
                        magmaFloatComplex **AII );

magma_int_t
magma_cgesellpmv_mpk(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaFloatComplex alpha,
                    magmaFloatComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaFloatComplex *d_x,
                    magmaFloatComplex beta,
                    magmaFloatComplex *d_y );



magma_int_t
magma_cilu_c( magma_c_sparse_matrix A,
                    magma_c_sparse_matrix A_ELLDD );

magma_int_t
magma_cilu_s( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_ELLDD );

magma_int_t
magma_cilu_cs(    magma_c_sparse_matrix A,
                    magma_c_sparse_matrix A_ELLDD );

magma_int_t
magma_cilu_ss( magma_c_sparse_matrix A,
                    magma_c_sparse_matrix A_ELLDD );
magma_int_t
magma_cilu_st( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_ELLDD );

magma_int_t
magma_cicc_ss( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_ELLD );

magma_int_t
magma_cicc_cs( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_ELLDD );

magma_int_t
magma_cilu_csr( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_CSRCSC );

magma_int_t
magma_caic_csr_c( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_CSR );

magma_int_t
magma_caic_csr_s( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_CSR );

magma_int_t
magma_caic_csr_cs( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_CSR );


magma_int_t
magma_caic_csr_ss( magma_c_sparse_matrix A,
                 magma_c_sparse_matrix A_CSR );


magma_int_t
magma_cailu_csr_s( magma_c_sparse_matrix A_L,
                   magma_c_sparse_matrix A_U,
                   magma_c_sparse_matrix L,
                   magma_c_sparse_matrix U );

magma_int_t
magma_cailu_csr_s_debug( magma_c_sparse_matrix A_L,
                   magma_c_sparse_matrix A_U,
                   magma_c_sparse_matrix L,
                   magma_c_sparse_matrix U );
magma_int_t
magma_cailu_csr_s_gs( magma_c_sparse_matrix A_L,
                   magma_c_sparse_matrix A_U,
                   magma_c_sparse_matrix L,
                   magma_c_sparse_matrix U,
                   magmaFloatComplex omega );

magma_int_t
magma_cailu_csr_s_gslu( magma_c_sparse_matrix A_L,
                   magma_c_sparse_matrix A_U,
                   magma_c_sparse_matrix L,
                   magma_c_sparse_matrix U,
                   magmaFloatComplex omega );


magma_int_t
magma_ctrisv_l( magma_c_sparse_matrix A,
                magma_c_vector b,
                magma_c_vector *x );

magma_int_t
magma_ctrisv_r( magma_c_sparse_matrix A,
                magma_c_vector b,
                magma_c_vector *x );

magma_int_t
magma_ctrisv_l_nu( magma_c_sparse_matrix A,
                magma_c_vector b,
                magma_c_vector *x );

magma_int_t
magma_ctrisv_r_nu( magma_c_sparse_matrix A,
                magma_c_vector b,
                magma_c_vector *x );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_c
#endif /* MAGMASPARSE_C_H */
