/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magmablas_z_v2.h normal z -> c, Mon May  2 23:31:25 2016
*/

#ifndef MAGMABLAS_C_H
#define MAGMABLAS_C_H

#include "magma_types.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
#define magmablas_ctranspose_inplace         magmablas_ctranspose_inplace_q
#define magmablas_ctranspose_conj_inplace    magmablas_ctranspose_conj_inplace_q
#define magmablas_ctranspose                 magmablas_ctranspose_q
#define magmablas_ctranspose_conj            magmablas_ctranspose_conj_q
#define magmablas_cgetmatrix_transpose       magmablas_cgetmatrix_transpose_q
#define magmablas_csetmatrix_transpose       magmablas_csetmatrix_transpose_q


  /*
   * RBT-related functions
   */
#define magmablas_cprbt                      magmablas_cprbt_q
#define magmablas_cprbt_mv                   magmablas_cprbt_mv_q
#define magmablas_cprbt_mtv                  magmablas_cprbt_mtv_q

                                                 
  /*
   * Multi-GPU copy functions
   */
#define magma_cgetmatrix_1D_col_bcyclic      magma_cgetmatrix_1D_col_bcyclic_q
#define magma_csetmatrix_1D_col_bcyclic      magma_csetmatrix_1D_col_bcyclic_q
#define magma_cgetmatrix_1D_row_bcyclic      magma_cgetmatrix_1D_row_bcyclic_q
#define magma_csetmatrix_1D_row_bcyclic      magma_csetmatrix_1D_row_bcyclic_q


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
#define magmablas_cgeadd                     magmablas_cgeadd_q
#define magmablas_cgeadd2                    magmablas_cgeadd2_q
#define magmablas_clacpy                     magmablas_clacpy_q
#define magmablas_clacpy_conj                magmablas_clacpy_conj_q
#define magmablas_clacpy_sym_in              magmablas_clacpy_sym_in_q
#define magmablas_clacpy_sym_out             magmablas_clacpy_sym_out_q
#define magmablas_clange                     magmablas_clange_q
#define magmablas_clanhe                     magmablas_clanhe_q
#define magmablas_clansy                     magmablas_clansy_q
#define magmablas_clarfg                     magmablas_clarfg_q
#define magmablas_clascl                     magmablas_clascl_q
#define magmablas_clascl_2x2                 magmablas_clascl_2x2_q
#define magmablas_clascl2                    magmablas_clascl2_q
#define magmablas_clascl_diag                magmablas_clascl_diag_q
#define magmablas_claset                     magmablas_claset_q
#define magmablas_claset_band                magmablas_claset_band_q
#define magmablas_claswp                     magmablas_claswp_q
#define magmablas_claswp2                    magmablas_claswp2_q
#define magmablas_claswp_sym                 magmablas_claswp_sym_q
#define magmablas_claswpx                    magmablas_claswpx_q
#define magmablas_csymmetrize                magmablas_csymmetrize_q
#define magmablas_csymmetrize_tiles          magmablas_csymmetrize_tiles_q
#define magmablas_ctrtri_diag                magmablas_ctrtri_diag_q


  /*
   * to cleanup (alphabetical order)
   */
#define magmablas_scnrm2_adjust              magmablas_scnrm2_adjust_q
#define magmablas_scnrm2_check               magmablas_scnrm2_check_q
#define magmablas_scnrm2_cols                magmablas_scnrm2_cols_q
#define magmablas_scnrm2_row_check_adjust    magmablas_scnrm2_row_check_adjust_q
#define magma_clarfb_gpu                     magma_clarfb_gpu_q
#define magma_clarfb_gpu_gemm                magma_clarfb_gpu_gemm_q
#define magma_clarfbx_gpu                    magma_clarfbx_gpu_q
#define magma_clarfg_gpu                     magma_clarfg_gpu_q
#define magma_clarfgtx_gpu                   magma_clarfgtx_gpu_q
#define magma_clarfgx_gpu                    magma_clarfgx_gpu_q
#define magma_clarfx_gpu                     magma_clarfx_gpu_q


  /*
   * Level 1 BLAS (alphabetical order)
   */
#define magmablas_caxpycp                    magmablas_caxpycp_q
#define magmablas_cswap                      magmablas_cswap_q
#define magmablas_cswapblk                   magmablas_cswapblk_q
#define magmablas_cswapdblk                  magmablas_cswapdblk_q


  /*
   * Level 2 BLAS (alphabetical order)
   */
#define magmablas_cgemv                      magmablas_cgemv_q
#define magmablas_cgemv_conj                 magmablas_cgemv_conj_q
#define magmablas_chemv                      magmablas_chemv_q
#define magmablas_csymv                      magmablas_csymv_q


  /*
   * Level 3 BLAS (alphabetical order)
   */
#define magmablas_cgemm                      magmablas_cgemm_q
#define magmablas_cgemm_reduce               magmablas_cgemm_reduce_q
#define magmablas_chemm                      magmablas_chemm_q
#define magmablas_csymm                      magmablas_csymm_q
#define magmablas_csyr2k                     magmablas_csyr2k_q
#define magmablas_cher2k                     magmablas_cher2k_q
#define magmablas_csyrk                      magmablas_csyrk_q
#define magmablas_cherk                      magmablas_cherk_q
#define magmablas_ctrsm                      magmablas_ctrsm_q
#define magmablas_ctrsm_outofplace           magmablas_ctrsm_outofplace_q
#define magmablas_ctrsm_work                 magmablas_ctrsm_work_q


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

#undef magma_csetvector
#undef magma_cgetvector
#undef magma_ccopyvector

#define magma_csetvector                     magma_csetvector_q
#define magma_cgetvector                     magma_cgetvector_q
#define magma_ccopyvector                    magma_ccopyvector_q


// ========================================
// copying sub-matrices (contiguous columns)

#undef magma_csetmatrix
#undef magma_cgetmatrix
#undef magma_ccopymatrix

#define magma_csetmatrix                     magma_csetmatrix_q
#define magma_cgetmatrix                     magma_cgetmatrix_q
#define magma_ccopymatrix                    magma_ccopymatrix_q


// ========================================
// Level 1 BLAS (alphabetical order)

#define magma_icamax                         magma_icamax_q
#define magma_icamin                         magma_icamin_q
#define magma_scasum                         magma_scasum_q
#define magma_caxpy                          magma_caxpy_q           
#define magma_ccopy                          magma_ccopy_q
#define magma_cdotc                          magma_cdotc_q
#define magma_cdotu                          magma_cdotu_q
#define magma_scnrm2                         magma_scnrm2_q
#define magma_crot                           magma_crot_q
#define magma_csrot                          magma_csrot_q
                        
#ifdef REAL             
#define magma_crotm                          magma_crotm_q
#define magma_crotmg                         magma_crotmg_q
#endif                  
                        
#define magma_cscal                          magma_cscal_q
#define magma_csscal                         magma_csscal_q
#define magma_cswap                          magma_cswap_q


// ========================================
// Level 2 BLAS (alphabetical order)

#define magma_cgemv                          magma_cgemv_q
#define magma_cgerc                          magma_cgerc_q
#define magma_cgeru                          magma_cgeru_q
#define magma_chemv                          magma_chemv_q
#define magma_cher                           magma_cher_q
#define magma_cher2                          magma_cher2_q
#define magma_ctrmv                          magma_ctrmv_q
#define magma_ctrsv                          magma_ctrsv_q


// ========================================
// Level 3 BLAS (alphabetical order)

#define magma_cgemm                          magma_cgemm_q
#define magma_csymm                          magma_csymm_q
#define magma_chemm                          magma_chemm_q
#define magma_csyr2k                         magma_csyr2k_q
#define magma_cher2k                         magma_cher2k_q
#define magma_csyrk                          magma_csyrk_q
#define magma_cherk                          magma_cherk_q
#define magma_ctrmm                          magma_ctrmm_q
#define magma_ctrsm                          magma_ctrsm_q

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMABLAS_C_H */
