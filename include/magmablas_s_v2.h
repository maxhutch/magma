/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magmablas_z_v2.h normal z -> s, Mon May  2 23:31:25 2016
*/

#ifndef MAGMABLAS_S_H
#define MAGMABLAS_S_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
#define magmablas_stranspose_inplace         magmablas_stranspose_inplace_q
#define magmablas_stranspose_inplace    magmablas_stranspose_inplace_q
#define magmablas_stranspose                 magmablas_stranspose_q
#define magmablas_stranspose            magmablas_stranspose_q
#define magmablas_sgetmatrix_transpose       magmablas_sgetmatrix_transpose_q
#define magmablas_ssetmatrix_transpose       magmablas_ssetmatrix_transpose_q


  /*
   * RBT-related functions
   */
#define magmablas_sprbt                      magmablas_sprbt_q
#define magmablas_sprbt_mv                   magmablas_sprbt_mv_q
#define magmablas_sprbt_mtv                  magmablas_sprbt_mtv_q

                                                 
  /*
   * Multi-GPU copy functions
   */
#define magma_sgetmatrix_1D_col_bcyclic      magma_sgetmatrix_1D_col_bcyclic_q
#define magma_ssetmatrix_1D_col_bcyclic      magma_ssetmatrix_1D_col_bcyclic_q
#define magma_sgetmatrix_1D_row_bcyclic      magma_sgetmatrix_1D_row_bcyclic_q
#define magma_ssetmatrix_1D_row_bcyclic      magma_ssetmatrix_1D_row_bcyclic_q


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
#define magmablas_sgeadd                     magmablas_sgeadd_q
#define magmablas_sgeadd2                    magmablas_sgeadd2_q
#define magmablas_slacpy                     magmablas_slacpy_q
#define magmablas_slacpy_conj                magmablas_slacpy_conj_q
#define magmablas_slacpy_sym_in              magmablas_slacpy_sym_in_q
#define magmablas_slacpy_sym_out             magmablas_slacpy_sym_out_q
#define magmablas_slange                     magmablas_slange_q
#define magmablas_slansy                     magmablas_slansy_q
#define magmablas_slansy                     magmablas_slansy_q
#define magmablas_slarfg                     magmablas_slarfg_q
#define magmablas_slascl                     magmablas_slascl_q
#define magmablas_slascl_2x2                 magmablas_slascl_2x2_q
#define magmablas_slascl2                    magmablas_slascl2_q
#define magmablas_slascl_diag                magmablas_slascl_diag_q
#define magmablas_slaset                     magmablas_slaset_q
#define magmablas_slaset_band                magmablas_slaset_band_q
#define magmablas_slaswp                     magmablas_slaswp_q
#define magmablas_slaswp2                    magmablas_slaswp2_q
#define magmablas_slaswp_sym                 magmablas_slaswp_sym_q
#define magmablas_slaswpx                    magmablas_slaswpx_q
#define magmablas_ssymmetrize                magmablas_ssymmetrize_q
#define magmablas_ssymmetrize_tiles          magmablas_ssymmetrize_tiles_q
#define magmablas_strtri_diag                magmablas_strtri_diag_q


  /*
   * to cleanup (alphabetical order)
   */
#define magmablas_snrm2_adjust              magmablas_snrm2_adjust_q
#define magmablas_snrm2_check               magmablas_snrm2_check_q
#define magmablas_snrm2_cols                magmablas_snrm2_cols_q
#define magmablas_snrm2_row_check_adjust    magmablas_snrm2_row_check_adjust_q
#define magma_slarfb_gpu                     magma_slarfb_gpu_q
#define magma_slarfb_gpu_gemm                magma_slarfb_gpu_gemm_q
#define magma_slarfbx_gpu                    magma_slarfbx_gpu_q
#define magma_slarfg_gpu                     magma_slarfg_gpu_q
#define magma_slarfgtx_gpu                   magma_slarfgtx_gpu_q
#define magma_slarfgx_gpu                    magma_slarfgx_gpu_q
#define magma_slarfx_gpu                     magma_slarfx_gpu_q


  /*
   * Level 1 BLAS (alphabetical order)
   */
#define magmablas_saxpycp                    magmablas_saxpycp_q
#define magmablas_sswap                      magmablas_sswap_q
#define magmablas_sswapblk                   magmablas_sswapblk_q
#define magmablas_sswapdblk                  magmablas_sswapdblk_q


  /*
   * Level 2 BLAS (alphabetical order)
   */
#define magmablas_sgemv                      magmablas_sgemv_q
#define magmablas_sgemv_conj                 magmablas_sgemv_conj_q
#define magmablas_ssymv                      magmablas_ssymv_q
#define magmablas_ssymv                      magmablas_ssymv_q


  /*
   * Level 3 BLAS (alphabetical order)
   */
#define magmablas_sgemm                      magmablas_sgemm_q
#define magmablas_sgemm_reduce               magmablas_sgemm_reduce_q
#define magmablas_ssymm                      magmablas_ssymm_q
#define magmablas_ssymm                      magmablas_ssymm_q
#define magmablas_ssyr2k                     magmablas_ssyr2k_q
#define magmablas_ssyr2k                     magmablas_ssyr2k_q
#define magmablas_ssyrk                      magmablas_ssyrk_q
#define magmablas_ssyrk                      magmablas_ssyrk_q
#define magmablas_strsm                      magmablas_strsm_q
#define magmablas_strsm_outofplace           magmablas_strsm_outofplace_q
#define magmablas_strsm_work                 magmablas_strsm_work_q


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

#undef magma_ssetvector
#undef magma_sgetvector
#undef magma_scopyvector

#define magma_ssetvector                     magma_ssetvector_q
#define magma_sgetvector                     magma_sgetvector_q
#define magma_scopyvector                    magma_scopyvector_q


// ========================================
// copying sub-matrices (contiguous columns)

#undef magma_ssetmatrix
#undef magma_sgetmatrix
#undef magma_scopymatrix

#define magma_ssetmatrix                     magma_ssetmatrix_q
#define magma_sgetmatrix                     magma_sgetmatrix_q
#define magma_scopymatrix                    magma_scopymatrix_q


// ========================================
// Level 1 BLAS (alphabetical order)

#define magma_isamax                         magma_isamax_q
#define magma_isamin                         magma_isamin_q
#define magma_sasum                         magma_sasum_q
#define magma_saxpy                          magma_saxpy_q           
#define magma_scopy                          magma_scopy_q
#define magma_sdot                          magma_sdot_q
#define magma_sdot                          magma_sdot_q
#define magma_snrm2                         magma_snrm2_q
#define magma_srot                           magma_srot_q
#define magma_srot                          magma_srot_q
                        
#ifdef REAL             
#define magma_srotm                          magma_srotm_q
#define magma_srotmg                         magma_srotmg_q
#endif                  
                        
#define magma_sscal                          magma_sscal_q
#define magma_sscal                         magma_sscal_q
#define magma_sswap                          magma_sswap_q


// ========================================
// Level 2 BLAS (alphabetical order)

#define magma_sgemv                          magma_sgemv_q
#define magma_sger                          magma_sger_q
#define magma_sger                          magma_sger_q
#define magma_ssymv                          magma_ssymv_q
#define magma_ssyr                           magma_ssyr_q
#define magma_ssyr2                          magma_ssyr2_q
#define magma_strmv                          magma_strmv_q
#define magma_strsv                          magma_strsv_q


// ========================================
// Level 3 BLAS (alphabetical order)

#define magma_sgemm                          magma_sgemm_q
#define magma_ssymm                          magma_ssymm_q
#define magma_ssymm                          magma_ssymm_q
#define magma_ssyr2k                         magma_ssyr2k_q
#define magma_ssyr2k                         magma_ssyr2k_q
#define magma_ssyrk                          magma_ssyrk_q
#define magma_ssyrk                          magma_ssyrk_q
#define magma_strmm                          magma_strmm_q
#define magma_strsm                          magma_strsm_q

#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMABLAS_S_H */
