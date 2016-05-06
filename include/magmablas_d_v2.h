/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magmablas_z_v2.h normal z -> d, Mon May  2 23:31:25 2016
*/

#ifndef MAGMABLAS_D_H
#define MAGMABLAS_D_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
#define magmablas_dtranspose_inplace         magmablas_dtranspose_inplace_q
#define magmablas_dtranspose_inplace    magmablas_dtranspose_inplace_q
#define magmablas_dtranspose                 magmablas_dtranspose_q
#define magmablas_dtranspose            magmablas_dtranspose_q
#define magmablas_dgetmatrix_transpose       magmablas_dgetmatrix_transpose_q
#define magmablas_dsetmatrix_transpose       magmablas_dsetmatrix_transpose_q


  /*
   * RBT-related functions
   */
#define magmablas_dprbt                      magmablas_dprbt_q
#define magmablas_dprbt_mv                   magmablas_dprbt_mv_q
#define magmablas_dprbt_mtv                  magmablas_dprbt_mtv_q

                                                 
  /*
   * Multi-GPU copy functions
   */
#define magma_dgetmatrix_1D_col_bcyclic      magma_dgetmatrix_1D_col_bcyclic_q
#define magma_dsetmatrix_1D_col_bcyclic      magma_dsetmatrix_1D_col_bcyclic_q
#define magma_dgetmatrix_1D_row_bcyclic      magma_dgetmatrix_1D_row_bcyclic_q
#define magma_dsetmatrix_1D_row_bcyclic      magma_dsetmatrix_1D_row_bcyclic_q


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
#define magmablas_dgeadd                     magmablas_dgeadd_q
#define magmablas_dgeadd2                    magmablas_dgeadd2_q
#define magmablas_dlacpy                     magmablas_dlacpy_q
#define magmablas_dlacpy_conj                magmablas_dlacpy_conj_q
#define magmablas_dlacpy_sym_in              magmablas_dlacpy_sym_in_q
#define magmablas_dlacpy_sym_out             magmablas_dlacpy_sym_out_q
#define magmablas_dlange                     magmablas_dlange_q
#define magmablas_dlansy                     magmablas_dlansy_q
#define magmablas_dlansy                     magmablas_dlansy_q
#define magmablas_dlarfg                     magmablas_dlarfg_q
#define magmablas_dlascl                     magmablas_dlascl_q
#define magmablas_dlascl_2x2                 magmablas_dlascl_2x2_q
#define magmablas_dlascl2                    magmablas_dlascl2_q
#define magmablas_dlascl_diag                magmablas_dlascl_diag_q
#define magmablas_dlaset                     magmablas_dlaset_q
#define magmablas_dlaset_band                magmablas_dlaset_band_q
#define magmablas_dlaswp                     magmablas_dlaswp_q
#define magmablas_dlaswp2                    magmablas_dlaswp2_q
#define magmablas_dlaswp_sym                 magmablas_dlaswp_sym_q
#define magmablas_dlaswpx                    magmablas_dlaswpx_q
#define magmablas_dsymmetrize                magmablas_dsymmetrize_q
#define magmablas_dsymmetrize_tiles          magmablas_dsymmetrize_tiles_q
#define magmablas_dtrtri_diag                magmablas_dtrtri_diag_q


  /*
   * to cleanup (alphabetical order)
   */
#define magmablas_dnrm2_adjust              magmablas_dnrm2_adjust_q
#define magmablas_dnrm2_check               magmablas_dnrm2_check_q
#define magmablas_dnrm2_cols                magmablas_dnrm2_cols_q
#define magmablas_dnrm2_row_check_adjust    magmablas_dnrm2_row_check_adjust_q
#define magma_dlarfb_gpu                     magma_dlarfb_gpu_q
#define magma_dlarfb_gpu_gemm                magma_dlarfb_gpu_gemm_q
#define magma_dlarfbx_gpu                    magma_dlarfbx_gpu_q
#define magma_dlarfg_gpu                     magma_dlarfg_gpu_q
#define magma_dlarfgtx_gpu                   magma_dlarfgtx_gpu_q
#define magma_dlarfgx_gpu                    magma_dlarfgx_gpu_q
#define magma_dlarfx_gpu                     magma_dlarfx_gpu_q


  /*
   * Level 1 BLAS (alphabetical order)
   */
#define magmablas_daxpycp                    magmablas_daxpycp_q
#define magmablas_dswap                      magmablas_dswap_q
#define magmablas_dswapblk                   magmablas_dswapblk_q
#define magmablas_dswapdblk                  magmablas_dswapdblk_q


  /*
   * Level 2 BLAS (alphabetical order)
   */
#define magmablas_dgemv                      magmablas_dgemv_q
#define magmablas_dgemv_conj                 magmablas_dgemv_conj_q
#define magmablas_dsymv                      magmablas_dsymv_q
#define magmablas_dsymv                      magmablas_dsymv_q


  /*
   * Level 3 BLAS (alphabetical order)
   */
#define magmablas_dgemm                      magmablas_dgemm_q
#define magmablas_dgemm_reduce               magmablas_dgemm_reduce_q
#define magmablas_dsymm                      magmablas_dsymm_q
#define magmablas_dsymm                      magmablas_dsymm_q
#define magmablas_dsyr2k                     magmablas_dsyr2k_q
#define magmablas_dsyr2k                     magmablas_dsyr2k_q
#define magmablas_dsyrk                      magmablas_dsyrk_q
#define magmablas_dsyrk                      magmablas_dsyrk_q
#define magmablas_dtrsm                      magmablas_dtrsm_q
#define magmablas_dtrsm_outofplace           magmablas_dtrsm_outofplace_q
#define magmablas_dtrsm_work                 magmablas_dtrsm_work_q


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// ========================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#undef magma_dsetvector
#undef magma_dgetvector
#undef magma_dcopyvector

#define magma_dsetvector                     magma_dsetvector_q
#define magma_dgetvector                     magma_dgetvector_q
#define magma_dcopyvector                    magma_dcopyvector_q


// ========================================
// copying sub-matrices (contiguous columns)

#undef magma_dsetmatrix
#undef magma_dgetmatrix
#undef magma_dcopymatrix

#define magma_dsetmatrix                     magma_dsetmatrix_q
#define magma_dgetmatrix                     magma_dgetmatrix_q
#define magma_dcopymatrix                    magma_dcopymatrix_q


// ========================================
// Level 1 BLAS (alphabetical order)

#define magma_idamax                         magma_idamax_q
#define magma_idamin                         magma_idamin_q
#define magma_dasum                         magma_dasum_q
#define magma_daxpy                          magma_daxpy_q           
#define magma_dcopy                          magma_dcopy_q
#define magma_ddot                          magma_ddot_q
#define magma_ddot                          magma_ddot_q
#define magma_dnrm2                         magma_dnrm2_q
#define magma_drot                           magma_drot_q
#define magma_drot                          magma_drot_q
                        
#ifdef REAL             
#define magma_drotm                          magma_drotm_q
#define magma_drotmg                         magma_drotmg_q
#endif                  
                        
#define magma_dscal                          magma_dscal_q
#define magma_dscal                         magma_dscal_q
#define magma_dswap                          magma_dswap_q


// ========================================
// Level 2 BLAS (alphabetical order)

#define magma_dgemv                          magma_dgemv_q
#define magma_dger                          magma_dger_q
#define magma_dger                          magma_dger_q
#define magma_dsymv                          magma_dsymv_q
#define magma_dsyr                           magma_dsyr_q
#define magma_dsyr2                          magma_dsyr2_q
#define magma_dtrmv                          magma_dtrmv_q
#define magma_dtrsv                          magma_dtrsv_q


// ========================================
// Level 3 BLAS (alphabetical order)

#define magma_dgemm                          magma_dgemm_q
#define magma_dsymm                          magma_dsymm_q
#define magma_dsymm                          magma_dsymm_q
#define magma_dsyr2k                         magma_dsyr2k_q
#define magma_dsyr2k                         magma_dsyr2k_q
#define magma_dsyrk                          magma_dsyrk_q
#define magma_dsyrk                          magma_dsyrk_q
#define magma_dtrmm                          magma_dtrmm_q
#define magma_dtrsm                          magma_dtrsm_q

#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMABLAS_D_H */
