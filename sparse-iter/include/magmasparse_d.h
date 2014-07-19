/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magmasparse_z.h normal z -> d, Fri Jul 18 17:34:26 2014
       @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_D_H
#define MAGMASPARSE_D_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_d


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Matrix Descriptors
*/
/* CSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   m;
    magma_int_t   n;
    magma_int_t nnz;

    double *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_dmatrix_t;


/* BCSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   rows_block;
    magma_int_t   cols_block;

    magma_int_t nrow_blocks;
    magma_int_t  nnz_blocks;

    double *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_dbcsr_t;


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/


magma_int_t 
read_d_csr_from_binary( magma_int_t* n_row, 
                        magma_int_t* n_col, 
                        magma_int_t* nnz, 
                        double **val, 
                        magma_index_t **row, 
                        magma_index_t **col,
                        const char * filename);

magma_int_t 
read_d_csr_from_mtx(    magma_storage_t *type, 
                        magma_location_t *location,
                        magma_int_t* n_row, 
                        magma_int_t* n_col, 
                        magma_int_t* nnz, 
                        double **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        const char *filename);

magma_int_t 
magma_d_csr_mtx(        magma_d_sparse_matrix *A, 
                        const char *filename );

magma_int_t 
magma_d_csr_mtxsymm(    magma_d_sparse_matrix *A, 
                        const char *filename );

magma_int_t 
magma_d_csr_compressor( double ** val, 
                        magma_index_t ** row, 
                        magma_index_t ** col, 
                        double ** valn, 
                        magma_index_t ** rown, 
                        magma_index_t ** coln, 
                        magma_int_t *n );

magma_int_t 
magma_d_csrtranspose(   magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *B );

magma_int_t 
magma_d_cucsrtranspose( magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *B );

magma_int_t 
d_transpose_csr(        magma_int_t n_rows, 
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
                        magma_index_t **new_col );

magma_int_t
magma_dcsrsplit(        magma_int_t bsize,
                        magma_d_sparse_matrix A,
                        magma_d_sparse_matrix *D,
                        magma_d_sparse_matrix *R );

magma_int_t
magma_dmscale(          magma_d_sparse_matrix *A, 
                        magma_scale_t scaling );

magma_int_t
magma_dmdiagadd(        magma_d_sparse_matrix *A, 
                        double add );

magma_int_t 
magma_dmsort(           magma_d_sparse_matrix *A );

magma_int_t
magma_dmhom(            magma_d_sparse_matrix A, 
                        magma_int_t b,  
                        magma_index_t *p );

magma_int_t
magma_dmhom_fd(         magma_d_sparse_matrix A, 
                        magma_int_t n,
                        magma_int_t b,  
                        magma_index_t *p );

magma_int_t 
magma_dilustruct(       magma_d_sparse_matrix *A,
                        magma_int_t levels );

magma_int_t 
magma_d_mpkinfo_one(    magma_d_sparse_matrix A, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s,    
                        magma_index_t **num_add_rows,
                        magma_index_t **add_rows,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs );

magma_int_t 
magma_d_mpkback(        magma_d_sparse_matrix A, 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s,
                        magma_int_t *num_add_vecs,
                        magma_index_t **add_vecs,
                        magma_int_t *num_vecs_back,
                        magma_index_t **vecs_back );

magma_int_t 
magma_d_mpkinfo(        magma_d_sparse_matrix A, 
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
magma_d_mpksetup_one(   magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *B, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s );

magma_int_t 
magma_d_mpksetup(       magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix B[MagmaMaxGPUs], 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s );

magma_int_t 
magma_d_mpk_compress(   magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        double *x,
                        double *y );

magma_int_t 
magma_d_mpk_uncompress( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        double *x,
                        double *y );

magma_int_t 
magma_d_mpk_uncompress_sel( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        magma_int_t offset,
                        magma_int_t blocksize,
                        double *x,
                        double *y );

magma_int_t 
magma_d_mpk_compress_gpu( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        double *x,
                        double *y );

magma_int_t 
magma_d_mpk_uncompress_gpu( magma_int_t num_add_rows,
                        magma_index_t *add_rows,
                        double *x,
                        double *y );

magma_int_t 
magma_d_mpk_uncompspmv(  magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows,
                         double *x,
                         double *y );

magma_int_t
magma_d_mpk_mcompresso(  magma_d_sparse_matrix A,
                         magma_d_sparse_matrix *B,
                         magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows );

magma_int_t 
write_d_csr_mtx(        magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        double **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        magma_order_t MajorType,
                        const char *filename );

magma_int_t 
write_d_csrtomtx(        magma_d_sparse_matrix A,
                        const char *filename );

magma_int_t 
print_d_csr(            magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        double **val, 
                        magma_index_t **row, 
                        magma_index_t **col );

magma_int_t 
print_d_csr_mtx(        magma_int_t n_row, 
                        magma_int_t n_col, 
                        magma_int_t nnz, 
                        double **val, 
                        magma_index_t **row, 
                        magma_index_t **col, 
                        magma_order_t MajorType );





magma_int_t 
magma_d_mtranspose(     magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *B );


magma_int_t 
magma_d_mtransfer(      magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *B, 
                        magma_location_t src, 
                        magma_location_t dst );

magma_int_t 
magma_d_vtransfer(      magma_d_vector x, 
                        magma_d_vector *y, 
                        magma_location_t src, 
                        magma_location_t dst );

magma_int_t 
magma_d_mconvert(       magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *B, 
                        magma_storage_t old_format, 
                        magma_storage_t new_format );

magma_int_t 
magma_d_LUmerge(        magma_d_sparse_matrix L, 
                        magma_d_sparse_matrix U, 
                        magma_d_sparse_matrix *B );

magma_int_t 
magma_d_LUmergein(      magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix L,
                        magma_d_sparse_matrix *B );

magma_int_t
magma_d_vinit(          magma_d_vector *x, 
                        magma_location_t memory_location,
                        magma_int_t num_rows, 
                        double values );

magma_int_t
magma_d_vvisu(          magma_d_vector x, 
                        magma_int_t offset, 
                        magma_int_t displaylength );

magma_int_t
magma_d_vread(          magma_d_vector *x, 
                        magma_int_t length,
                        char * filename );
magma_int_t
magma_d_mvisu(          magma_d_sparse_matrix A );

magma_int_t 
magma_ddiameter(        magma_d_sparse_matrix *A );

magma_int_t 
magma_drowentries(      magma_d_sparse_matrix *A );

magma_int_t
magma_d_mfree(          magma_d_sparse_matrix *A );

magma_int_t
magma_d_vfree(          magma_d_vector *x );

magma_int_t
magma_dresidual(        magma_d_sparse_matrix A, 
                        magma_d_vector b, 
                        magma_d_vector x, 
                        double *res );

magma_int_t
magma_dbitflip(     double *d, 
                    magma_int_t loc, 
                    magma_int_t bit );

magma_int_t
magma_dmgenerator(  magma_int_t n,
                    magma_int_t offdiags,
                    magma_index_t *diag_offset,
                    double *diag_vals,
                    magma_d_sparse_matrix *A );

magma_int_t
magma_dsolverinfo(  magma_d_solver_par *solver_par, 
                    magma_d_preconditioner *precond_par );

magma_int_t
magma_dsolverinfo_init( magma_d_solver_par *solver_par, 
                        magma_d_preconditioner *precond );

magma_int_t
magma_dsolverinfo_free( magma_d_solver_par *solver_par, 
                        magma_d_preconditioner *precond );


magma_int_t 
magma_dfrobenius( magma_d_sparse_matrix A, magma_d_sparse_matrix B, 
                  real_Double_t *res );

magma_int_t 
magma_dnonlinres(   magma_d_sparse_matrix A, 
                    magma_d_sparse_matrix L,
                    magma_d_sparse_matrix U, 
                    magma_d_sparse_matrix *LU, 
                    real_Double_t *res );

magma_int_t 
magma_dilures(      magma_d_sparse_matrix A, 
                    magma_d_sparse_matrix L,
                    magma_d_sparse_matrix U, 
                    magma_d_sparse_matrix *LU, 
                    real_Double_t *res );

magma_int_t 
magma_dinitguess(   magma_d_sparse_matrix A, 
                    magma_d_sparse_matrix *L,
                    magma_d_sparse_matrix *U );

magma_int_t
magma_dmreorder( magma_d_sparse_matrix A, magma_int_t n, magma_int_t b, magma_d_sparse_matrix *B );

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
magma_dcg(             magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t 
magma_dcg_res(         magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t 
magma_dcg_sdc(         magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t 
magma_dcg_merge(       magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dcg_exactres(    magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t 
magma_dgmres(          magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dbicgstab(       magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
                       magma_d_solver_par *solver_par );

magma_int_t
magma_dbicgstab_merge( magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dbicgstab_merge2( magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dpcg(            magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par, 
                       magma_d_preconditioner *precond_par );

magma_int_t
magma_dpbicgstab(      magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par, 
                       magma_d_preconditioner *precond_par );

magma_int_t
magma_dpgmres(         magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par, 
                       magma_d_preconditioner *precond_par );
magma_int_t
magma_djacobi(         magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dbaiter(         magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_diterref(        magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par, 
                       magma_d_preconditioner *precond_par );
magma_int_t
magma_dp1gmres(        magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dgmres_pipe(     magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dilu(            magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par, 
                       magma_int_t *ipiv );

magma_int_t
magma_dbcsrlu(         magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );


magma_int_t
magma_dbcsrlutrf(      magma_d_sparse_matrix A, 
                       magma_d_sparse_matrix *M,
                       magma_int_t *ipiv, 
                       magma_int_t version );

magma_int_t
magma_dbcsrlusv(       magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par, 
                       magma_int_t *ipiv );



magma_int_t
magma_dilucg(          magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par );

magma_int_t
magma_dilugmres(       magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par ); 


magma_int_t
magma_dlobpcg_shift(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magma_int_t shift,
                        double *x );
magma_int_t
magma_dlobpcg_res(      magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        double *evalues, 
                        double *X,
                        double *R, 
                        double *res );
magma_int_t
magma_dlobpcg_maxpy(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        double *X,
                        double *Y);

/*/////////////////////////////////////////////////////////////////////////////
    -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_dlobpcg(          magma_d_sparse_matrix A,
                        magma_d_solver_par *solver_par );




/*/////////////////////////////////////////////////////////////////////////////
    -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_djacobisetup(     magma_d_sparse_matrix A, 
                        magma_d_vector b, 
                        magma_d_sparse_matrix *M, 
                        magma_d_vector *c );
magma_int_t
magma_djacobisetup_matrix(  magma_d_sparse_matrix A, 
                            magma_d_vector b, 
                            magma_d_sparse_matrix *M, 
                            magma_d_vector *d );
magma_int_t
magma_djacobisetup_vector(  magma_d_vector b,  
                            magma_d_vector d, 
                            magma_d_vector *c );

magma_int_t
magma_djacobiiter(      magma_d_sparse_matrix M, 
                        magma_d_vector c, 
                        magma_d_vector *x,  
                        magma_d_solver_par *solver_par );

magma_int_t
magma_dilusetup(        magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix *M,
                        magma_int_t *ipiv );

magma_int_t
magma_dpastixsetup(     magma_d_sparse_matrix A, magma_d_vector b,
                        magma_d_preconditioner *precond );


magma_int_t
magma_dapplypastix(     magma_d_vector b, magma_d_vector *x, 
                        magma_d_preconditioner *precond );


// CUSPARSE preconditioner

magma_int_t
magma_dcuilusetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond );

magma_int_t
magma_dapplycuilu_l( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond );
magma_int_t
magma_dapplycuilu_r( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond );


magma_int_t
magma_dcuiccsetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond );

magma_int_t
magma_dapplycuicc_l( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond );
magma_int_t
magma_dapplycuicc_r( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond );

// block-asynchronous preconditioner

magma_int_t
magma_dailusetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond );

magma_int_t
magma_daiccsetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond );

magma_int_t
magma_dapplyailu_l( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond );
magma_int_t
magma_dapplyailu_r( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond );


// block-asynchronous iteration

magma_int_t
magma_dbajac_csr(   magma_int_t localiters,
                    magma_d_sparse_matrix D,
                    magma_d_sparse_matrix R,
                    magma_d_vector b,
                    magma_d_vector *x );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_d_spmv(           double alpha, 
                        magma_d_sparse_matrix A, 
                        magma_d_vector x, 
                        double beta, 
                        magma_d_vector y );

magma_int_t
magma_d_spmv_shift(     double alpha, 
                        magma_d_sparse_matrix A, 
                        double lambda,
                        magma_d_vector x, 
                        double beta, 
                        magma_int_t offset,     
                        magma_int_t blocksize,
                        magma_index_t *add_vecs, 
                        magma_d_vector y );

magma_int_t
magma_dcuspmm(          magma_d_sparse_matrix A, 
                        magma_d_sparse_matrix B, 
                        magma_d_sparse_matrix *AB );

magma_int_t
magma_d_precond(        magma_d_sparse_matrix A, 
                        magma_d_vector b, magma_d_vector *x,
                        magma_d_preconditioner precond );

magma_int_t
magma_d_precondsetup( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_preconditioner *precond );

magma_int_t
magma_d_applyprecond( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_vector *x, magma_d_preconditioner *precond );


magma_int_t
magma_d_applyprecond_left( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_vector *x, magma_d_preconditioner *precond );


magma_int_t
magma_d_applyprecond_right( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_vector *x, magma_d_preconditioner *precond );

magma_int_t
magma_dorthomgs(        magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        double *X );

magma_int_t
magma_d_initP2P(        magma_int_t *bandwidth_benchmark,
                        magma_int_t *num_gpus );




/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_dgecsrmv(        magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       double *d_y );

magma_int_t 
magma_dgecsrmv_shift(  magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       double alpha,
                       double lambda,
                       double *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       double *d_y );

magma_int_t 
magma_dmgecsrmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_rowptr,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       double *d_y );

magma_int_t 
magma_dgeellmv(        magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       double *d_y );

magma_int_t 
magma_dgeellmv_shift(  magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double lambda,
                       double *d_val,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       double *d_y );


magma_int_t 
magma_dmgeellmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       double *d_y );


magma_int_t 
magma_dgeelltmv(       magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       double *d_y );

magma_int_t 
magma_dgeelltmv_shift( magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double lambda,
                       double *d_val,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       int offset,
                       int blocksize,
                       magma_index_t *add_rows,
                       double *d_y );


magma_int_t 
magma_dmgeelltmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_colind,
                       double *d_x,
                       double beta,
                       double *d_y );

magma_int_t 
magma_dgeellrtmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_colind,
                       magma_index_t *d_rowlength,
                       double *d_x,
                       double beta,
                       double *d_y,
                       magma_int_t num_threads,
                       magma_int_t threads_per_row );

magma_int_t 
magma_dgesellcmv(      magma_trans_t transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t blocksize,
                       magma_int_t slices,
                       magma_int_t alignment,
                       double alpha,
                       double *d_val,
                       magma_index_t *d_colind,
                       magma_index_t *d_rowptr,
                       double *d_x,
                       double beta,
                       double *d_y );

magma_int_t
magma_dgesellpmv(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    double alpha,
                    double *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    double *d_x,
                    double beta,
                    double *d_y );

magma_int_t
magma_dmgesellpmv( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    double alpha,
                    double *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    double *d_x,
                    double beta,
                    double *d_y );

magma_int_t
magma_dmgesellpmv_blocked( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    double alpha,
                    double *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    double *d_x,
                    double beta,
                    double *d_y );


magma_int_t 
magma_dp1gmres_mgs(    magma_int_t  n, 
                       magma_int_t  k, 
                       double *skp, 
                       double *v, 
                       double *z );


magma_int_t
magma_dmergedgs(        magma_int_t n, 
                        magma_int_t ldh,
                        magma_int_t k, 
                        double *v, 
                        double *r,
                        double *skp );

magma_int_t
magma_dcopyscale(       int n, 
                        int k,
                        double *r, 
                        double *v,
                        double *skp );

magma_int_t
magma_dnrm2scale(      int m, 
                        double *r, int lddr, 
                        double *drnorm);


magma_int_t
magma_djacobisetup_vector_gpu( int num_rows, 
                               double *b, 
                               double *d, 
                               double *c);

magma_int_t
magma_djacobi_diagscal(         int num_rows, 
                                double *b, 
                                double *d, 
                                double *c);

magma_int_t
magma_djacobisetup_diagscal( magma_d_sparse_matrix A, magma_d_vector *d );


magma_int_t
magma_dbicgmerge1(  int n, 
                    double *skp,
                    double *v, 
                    double *r, 
                    double *p );


magma_int_t
magma_dbicgmerge2(  int n, 
                    double *skp, 
                    double *r,
                    double *v, 
                    double *s );

magma_int_t
magma_dbicgmerge3(  int n, 
                    double *skp, 
                    double *p,
                    double *s,
                    double *t,
                    double *x, 
                    double *r );
magma_int_t
magma_dbicgmerge4(  int type, 
                    double *skp );

magma_int_t
magma_dcgmerge_spmv1(  
                 magma_d_sparse_matrix A,
                 double *d1,
                 double *d2,
                 double *d_d,
                 double *d_z,
                 double *skp );

magma_int_t
magma_dcgmerge_xrbeta(  
                 int n,
                 double *d1,
                 double *d2,
                 double *x,
                 double *r,
                 double *d,
                 double *z, 
                 double *skp );

magma_int_t
magma_dmdotc(       magma_int_t n, 
                    magma_int_t k, 
                    double *v, 
                    double *r,
                    double *d1,
                    double *d2,
                    double *skp );

magma_int_t
magma_dgemvmdot(    int n, 
                    int k, 
                    double *v, 
                    double *r,
                    double *d1,
                    double *d2,
                    double *skp );

magma_int_t
magma_dbicgmerge_spmv1(  
                 magma_d_sparse_matrix A,
                 double *d1,
                 double *d2,
                 double *d_p,
                 double *d_r,
                 double *d_v,
                 double *skp );

magma_int_t
magma_dbicgmerge_spmv2(  
                 magma_d_sparse_matrix A,
                 double *d1,
                 double *d2,
                 double *d_s,
                 double *d_t,
                 double *skp );

magma_int_t
magma_dbicgmerge_xrbeta(  
                 int n,
                 double *d1,
                 double *d2,
                 double *rr,
                 double *r,
                 double *p,
                 double *s,
                 double *t,
                 double *x, 
                 double *skp );

magma_int_t
magma_dbcsrswp(  magma_int_t n,
                 magma_int_t size_b, 
                 magma_int_t *ipiv,
                 double *x );

magma_int_t
magma_dbcsrtrsv( magma_uplo_t uplo,
                 magma_int_t r_blocks,
                 magma_int_t c_blocks,
                 magma_int_t size_b, 
                 double *A,
                 magma_index_t *blockinfo,   
                 double *x );

magma_int_t
magma_dbcsrvalcpy(  magma_int_t size_b, 
                    magma_int_t num_blocks, 
                    magma_int_t num_zero_blocks, 
                    double **Aval, 
                    double **Bval,
                    double **Bval2 );

magma_int_t
magma_dbcsrluegemm( magma_int_t size_b, 
                    magma_int_t num_block_rows,
                    magma_int_t kblocks,
                    double **dA,  
                    double **dB,  
                    double **dC );

magma_int_t
magma_dbcsrlupivloc( magma_int_t size_b, 
                    magma_int_t kblocks,
                    double **dA,  
                    magma_int_t *ipiv );

magma_int_t
magma_dbcsrblockinfo5(  magma_int_t lustep,
                        magma_int_t num_blocks, 
                        magma_int_t c_blocks, 
                        magma_int_t size_b,
                        magma_index_t *blockinfo,
                        double *val,
                        double **AII );

magma_int_t
magma_dgesellpmv_mpk(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    double alpha,
                    double *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    double *d_x,
                    double beta,
                    double *d_y );



magma_int_t
magma_dilu_c( magma_d_sparse_matrix A,
                    magma_d_sparse_matrix A_ELLDD );

magma_int_t
magma_dilu_s( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_ELLDD );

magma_int_t
magma_dilu_cs(    magma_d_sparse_matrix A,
                    magma_d_sparse_matrix A_ELLDD );

magma_int_t
magma_dilu_ss( magma_d_sparse_matrix A,
                    magma_d_sparse_matrix A_ELLDD );
magma_int_t
magma_dilu_st( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_ELLDD );

magma_int_t
magma_dicc_ss( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_ELLD );

magma_int_t
magma_dicc_cs( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_ELLDD );

magma_int_t
magma_dilu_csr( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_CSRCSC );

magma_int_t
magma_daic_csr_a( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_CSR );

magma_int_t
magma_daic_csr_s( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_CSR );

magma_int_t
magma_daic_csr_cs( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_CSR );


magma_int_t
magma_daic_csr_ss( magma_d_sparse_matrix A,
                 magma_d_sparse_matrix A_CSR );


magma_int_t
magma_dailu_csr_s( magma_d_sparse_matrix A_L,
                   magma_d_sparse_matrix A_U,
                   magma_d_sparse_matrix L,
                   magma_d_sparse_matrix U );

magma_int_t
magma_dailu_csr_a( magma_d_sparse_matrix A,
                   magma_d_sparse_matrix L,
                   magma_d_sparse_matrix U );

magma_int_t
magma_dailu_csr_s_debug( magma_index_t *p,
                   magma_d_sparse_matrix A_L,
                   magma_d_sparse_matrix A_U,
                   magma_d_sparse_matrix L,
                   magma_d_sparse_matrix U );
magma_int_t
magma_dailu_csr_s_gs( magma_d_sparse_matrix A_L,
                   magma_d_sparse_matrix A_U,
                   magma_d_sparse_matrix L,
                   magma_d_sparse_matrix U,
                   double omega );

magma_int_t
magma_dailu_csr_s_gslu( magma_d_sparse_matrix A_L,
                   magma_d_sparse_matrix A_U,
                   magma_d_sparse_matrix L,
                   magma_d_sparse_matrix U,
                   double omega );


magma_int_t
magma_dtrisv_l( magma_d_sparse_matrix A,
                magma_d_vector b,
                magma_d_vector *x );

magma_int_t
magma_dtrisv_r( magma_d_sparse_matrix A,
                magma_d_vector b,
                magma_d_vector *x );

magma_int_t
magma_dtrisv_l_nu( magma_d_sparse_matrix A,
                magma_d_vector b,
                magma_d_vector *x );

magma_int_t
magma_dtrisv_r_nu( magma_d_sparse_matrix A,
                magma_d_vector b,
                magma_d_vector *x );

magma_int_t 
mtools_dsymmetrize( magma_d_sparse_matrix *A );

 
#ifdef __cplusplus
}
#endif

#undef PRECISION_d
#endif /* MAGMASPARSE_D_H */
