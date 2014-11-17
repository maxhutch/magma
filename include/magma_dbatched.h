/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from magma_zbatched.h normal z -> d, Sat Nov 15 19:53:54 2014
*/

#ifndef MAGMA_DBATCHED_H
#define MAGMA_DBATCHED_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void 
dset_pointer_int(magma_int_t **output_array,
        magma_int_t *input,
        magma_int_t lda,
        magma_int_t row, magma_int_t column, 
        magma_int_t batchSize,
        magma_int_t batchCount);

void 
dset_pointer(double **output_array,
                 double *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column,
                 magma_int_t batchSize,
                 magma_int_t batchCount);


void 
dset_array(double **output_array,
               double **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column,
               magma_int_t batchCount);

void 
magma_ddisplace_pointers(double **output_array,
               double **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount);

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */

void
magmablas_dgemm_batched(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double const * const * dB_array, magma_int_t lddb,
    double beta,
    double **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_dgemm_batched_lg(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double const * const * dB_array, magma_int_t lddb,
    double beta,
    double **dC_array, magma_int_t lddc, magma_int_t batchCount );
void
magmablas_dgemm_batched_k32(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double const * const * dB_array, magma_int_t lddb,
    double beta,
    double **dC_array, magma_int_t lddc, magma_int_t batchCount );

void 
magmablas_dgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    double **dA_array, magma_int_t ldda,
    double **dx_array, magma_int_t incx,
    double beta,
    double **dy_array, magma_int_t incy,
    magma_int_t batchCount);

void 
magmablas_dsyrk_NC_batched( magma_trans_t TRANSA, magma_trans_t TRANSB, int m , int n , int k , 
                       double alpha, double **dA_array, int lda, 
                       double **B_array, int ldb, 
                       double beta,        double **C_array, int ldc, 
                       magma_int_t batchCount);

void
magmablas_dsyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double beta,
    double **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_dsyrk_batched_lg(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double beta,
    double **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_dsyrk_batched_k32(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double beta,
    double **dC_array, magma_int_t lddc, magma_int_t batchCount );






magma_int_t 
magma_dpotf2_tile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);

magma_int_t 
magma_dpotrf_panel(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    double *A, magma_int_t lda,
    double* dinvA, double** dinvA_array, magma_int_t invA_msize,
    double* x, double** x_array,  magma_int_t x_msize,
    magma_int_t *info_array,  magma_int_t batchCount, magma_int_t matrixSize);


void 
magmablas_dtrtri_diag_batched_q(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    double const * const *dA_array, magma_int_t ldda,
    double **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_dtrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    double const * const *dA_array, magma_int_t ldda,
    double **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount);


void 
magmablas_dtrsm_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    double** dA_array,    magma_int_t ldda,
    double** dB_array,    magma_int_t lddb,
    magma_int_t batchCount);


void 
magmablas_dtrsm_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    double alpha, 
    double** dA_array,    magma_int_t ldda,
    double** dB_array,    magma_int_t lddb,
    double** dX_array,    magma_int_t lddx, 
    double** dinvA_array, magma_int_t dinvA_length,
    double** dA_displ, double** dB_displ, 
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount);

void 
magmablas_dtrsm_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    double alpha, 
    double** dA_array,    magma_int_t ldda,
    double** dB_array,    magma_int_t lddb,
    double** dX_array,    magma_int_t lddx, 
    double** dinvA_array, magma_int_t dinvA_length,
    double** dA_displ, double** dB_displ, 
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount);

magma_int_t 
magma_dpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    double **dA_array, magma_int_t lda,
    magma_int_t *info_array,  magma_int_t batchCount);


magma_int_t 
magma_dpotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double **dA_array, magma_int_t lda,
    double **dA_displ, 
    double **dW_displ,
    double **dB_displ, 
    double **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_dpotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ, 
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_dpotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ,  
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_dpotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ,  
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_dpotrs_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  double **dB_array, magma_int_t lddb,
                  magma_int_t batchCount);

magma_int_t 
magma_dposv_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  double **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount);

magma_int_t 
magma_dgetrs_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  double **dB_array, magma_int_t lddb,
                  magma_int_t batchCount);



void 
magma_dlaswp_rowparallel_batched( magma_int_t n, double** input_array, magma_int_t ldi,
                   double** output_array, magma_int_t ldo,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **pivinfo_array, 
                   magma_int_t batchCount );

void 
magma_dlaswp_rowserial_batched(magma_int_t n, double** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount);

void 
magma_dlaswp_columnserial_batched(magma_int_t n, double** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount);

void 
magmablas_dtranspose_batched(
    magma_int_t m, magma_int_t n,
    double **dA_array,  magma_int_t ldda,
    double **dAT_array, magma_int_t lddat, magma_int_t batchCount );

void 
magmablas_dlaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount );

void 
magmablas_dlaset_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_dmemset_batched(magma_int_t length, 
        magmaDouble_ptr dAarray[], double val, 
        magma_int_t batchCount);

magma_int_t 
magma_dgetf2_batched(
    magma_int_t m, magma_int_t n,
    double **dA_array, magma_int_t lda,
    double **GERA_array,
    double **GERB_array,
    double **GERC_array,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, 
    magma_int_t gbstep,            
    magma_int_t batchCount,
    cublasHandle_t myhandle);

magma_int_t 
magma_dgetrf_recpanel_batched_q(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    double** dA_array,    magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW1_displ, double** dW2_displ,  
    double** dW3_displ, double** dW4_displ,
    double** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);

magma_int_t 
magma_dgetrf_batched(
        magma_int_t m, magma_int_t n,
        double **dA_array, 
        magma_int_t lda,
        magma_int_t **ipiv_array, 
        magma_int_t *info_array, 
        magma_int_t batchCount);

magma_int_t 
magma_dgetri_outofplace_batched( magma_int_t n, 
                  double **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  double **dinvA_array, magma_int_t lddia,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

void 
magma_ddisplace_intpointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount);





void 
magmablas_idamax_atomic_batched(magma_int_t n, double** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_idamax_tree_batched(magma_int_t n, double** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);



void 
magmablas_idamax_batched(magma_int_t n, double** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_idamax(magma_int_t n, double* x, magma_int_t incx, magma_int_t *max_id);


magma_int_t 
magma_idamax_batched(magma_int_t length, 
        double **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);

magma_int_t 
magma_dswap_batched(magma_int_t n, double **x_array, magma_int_t incx, magma_int_t j, 
                 magma_int_t** ipiv_array, magma_int_t batchCount);

magma_int_t 
magma_dscal_dger_batched(magma_int_t m, magma_int_t n, magma_int_t step,
                                      double **dA_array, magma_int_t lda,
                                      magma_int_t *info_array, magma_int_t gbstep, 
                                      magma_int_t batchCount);

magma_int_t 
magma_dcomputecolumn_batched(magma_int_t m, magma_int_t paneloffset, magma_int_t step, 
                                        double **dA_array,  magma_int_t lda,
                                        magma_int_t **ipiv_array, 
                                        magma_int_t *info_array, magma_int_t gbstep, 
                                        magma_int_t batchCount);

void 
magma_dgetf2trsm_batched(magma_int_t ib, magma_int_t n, double **dA_array,  magma_int_t j, magma_int_t lda,
                       magma_int_t batchCount);


magma_int_t 
magma_dgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    double **dA_array, magma_int_t lda,
    double **dW0_displ,
    double **dW1_displ,
    double **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount,
    cublasHandle_t myhandle);

magma_int_t 
magma_dgetrf_panel_nopiv_batched_q(
    magma_int_t m, magma_int_t nb,    
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ,  
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,     
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);

magma_int_t 
magma_dgetrf_recpanel_nopiv_batched_q(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW1_displ, double** dW2_displ,  
    double** dW3_displ, double** dW4_displ,
    double** dW5_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);


magma_int_t 
magma_dgetrf_nopiv_batched(
        magma_int_t m, magma_int_t n,
        double **dA_array, 
        magma_int_t lda,
        magma_int_t *info_array, 
        magma_int_t batchCount);

magma_int_t 
magma_dgetrs_nopiv_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  double **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_dgesv_nopiv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  double **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_dgesv_rbt_batched(
                  magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  double **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_dgesv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  double **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  double **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount);

magma_int_t
magma_dgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    double **dA_array, magma_int_t ldda,
    double **dB_array, magma_int_t lddb,
    double *U, double *V,
    magma_int_t *info, magma_int_t batchCount);

void 
magmablas_dprbt_batched(
    magma_int_t n, 
    double **dA_array, magma_int_t ldda, 
    double *du, double *dv,
    magma_int_t batchCount);

void
magmablas_dprbt_mv_batched(
    magma_int_t n, 
    double *dv, double **db_array, magma_int_t batchCount);


void
magmablas_dprbt_mtv_batched(
    magma_int_t n, 
    double *du, double **db_array, magma_int_t batchCount);





void 
magma_dlacgv_batched(magma_int_t n, double **x_array, magma_int_t incx, int offset, int batchCount);

void 
magma_dpotf2_dscal_batched(magma_int_t n, double **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t batchCount);

void 
magma_dpotf2_ddot_batched(magma_int_t n, double **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);


void 
setup_pivinfo_q( magma_int_t *pivinfo, magma_int_t *ipiv, 
                      magma_int_t m, magma_int_t nb, 
                      magma_queue_t stream);

void 
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, 
                    magma_int_t m, magma_int_t nb);


void
magmablas_dgeadd_batched_q(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDouble_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dlacpy_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDouble_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );
void
magmablas_dgeadd_batched(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDouble_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );

void
magmablas_dlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDouble_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );

void
magmablas_dgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t ldda,
    magmaDouble_ptr dx_array[], magma_int_t incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t incy,
    magma_int_t batchCount);


// for debugging purpose
void 
dset_stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount);



#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMA_DBATCHED_H */
