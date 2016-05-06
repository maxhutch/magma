/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c
*/

#ifndef MAGMABLAS_Z_H
#define MAGMABLAS_Z_H

#include "magma_types.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
#define magmablas_ztranspose_inplace         magmablas_ztranspose_inplace_q
#define magmablas_ztranspose_conj_inplace    magmablas_ztranspose_conj_inplace_q
#define magmablas_ztranspose                 magmablas_ztranspose_q
#define magmablas_ztranspose_conj            magmablas_ztranspose_conj_q
#define magmablas_zgetmatrix_transpose       magmablas_zgetmatrix_transpose_q
#define magmablas_zsetmatrix_transpose       magmablas_zsetmatrix_transpose_q


  /*
   * RBT-related functions
   */
#define magmablas_zprbt                      magmablas_zprbt_q
#define magmablas_zprbt_mv                   magmablas_zprbt_mv_q
#define magmablas_zprbt_mtv                  magmablas_zprbt_mtv_q

                                                 
  /*
   * Multi-GPU copy functions
   */
#define magma_zgetmatrix_1D_col_bcyclic      magma_zgetmatrix_1D_col_bcyclic_q
#define magma_zsetmatrix_1D_col_bcyclic      magma_zsetmatrix_1D_col_bcyclic_q
#define magma_zgetmatrix_1D_row_bcyclic      magma_zgetmatrix_1D_row_bcyclic_q
#define magma_zsetmatrix_1D_row_bcyclic      magma_zsetmatrix_1D_row_bcyclic_q


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
#define magmablas_zgeadd                     magmablas_zgeadd_q
#define magmablas_zgeadd2                    magmablas_zgeadd2_q
#define magmablas_zlacpy                     magmablas_zlacpy_q
#define magmablas_zlacpy_conj                magmablas_zlacpy_conj_q
#define magmablas_zlacpy_sym_in              magmablas_zlacpy_sym_in_q
#define magmablas_zlacpy_sym_out             magmablas_zlacpy_sym_out_q
#define magmablas_zlange                     magmablas_zlange_q
#define magmablas_zlanhe                     magmablas_zlanhe_q
#define magmablas_zlansy                     magmablas_zlansy_q
#define magmablas_zlarfg                     magmablas_zlarfg_q
#define magmablas_zlascl                     magmablas_zlascl_q
#define magmablas_zlascl_2x2                 magmablas_zlascl_2x2_q
#define magmablas_zlascl2                    magmablas_zlascl2_q
#define magmablas_zlascl_diag                magmablas_zlascl_diag_q
#define magmablas_zlaset                     magmablas_zlaset_q
#define magmablas_zlaset_band                magmablas_zlaset_band_q
#define magmablas_zlaswp                     magmablas_zlaswp_q
#define magmablas_zlaswp2                    magmablas_zlaswp2_q
#define magmablas_zlaswp_sym                 magmablas_zlaswp_sym_q
#define magmablas_zlaswpx                    magmablas_zlaswpx_q
#define magmablas_zsymmetrize                magmablas_zsymmetrize_q
#define magmablas_zsymmetrize_tiles          magmablas_zsymmetrize_tiles_q
#define magmablas_ztrtri_diag                magmablas_ztrtri_diag_q


  /*
   * to cleanup (alphabetical order)
   */
#define magmablas_dznrm2_adjust              magmablas_dznrm2_adjust_q
#define magmablas_dznrm2_check               magmablas_dznrm2_check_q
#define magmablas_dznrm2_cols                magmablas_dznrm2_cols_q
#define magmablas_dznrm2_row_check_adjust    magmablas_dznrm2_row_check_adjust_q
#define magma_zlarfb_gpu                     magma_zlarfb_gpu_q
#define magma_zlarfb_gpu_gemm                magma_zlarfb_gpu_gemm_q
#define magma_zlarfbx_gpu                    magma_zlarfbx_gpu_q
#define magma_zlarfg_gpu                     magma_zlarfg_gpu_q
#define magma_zlarfgtx_gpu                   magma_zlarfgtx_gpu_q
#define magma_zlarfgx_gpu                    magma_zlarfgx_gpu_q
#define magma_zlarfx_gpu                     magma_zlarfx_gpu_q


  /*
   * Level 1 BLAS (alphabetical order)
   */
#define magmablas_zaxpycp                    magmablas_zaxpycp_q
#define magmablas_zswap                      magmablas_zswap_q
#define magmablas_zswapblk                   magmablas_zswapblk_q
#define magmablas_zswapdblk                  magmablas_zswapdblk_q


  /*
   * Level 2 BLAS (alphabetical order)
   */
#define magmablas_zgemv                      magmablas_zgemv_q
#define magmablas_zgemv_conj                 magmablas_zgemv_conj_q
#define magmablas_zhemv                      magmablas_zhemv_q
#define magmablas_zsymv                      magmablas_zsymv_q


  /*
   * Level 3 BLAS (alphabetical order)
   */
#define magmablas_zgemm                      magmablas_zgemm_q
#define magmablas_zgemm_reduce               magmablas_zgemm_reduce_q
#define magmablas_zhemm                      magmablas_zhemm_q
#define magmablas_zsymm                      magmablas_zsymm_q
#define magmablas_zsyr2k                     magmablas_zsyr2k_q
#define magmablas_zher2k                     magmablas_zher2k_q
#define magmablas_zsyrk                      magmablas_zsyrk_q
#define magmablas_zherk                      magmablas_zherk_q
#define magmablas_ztrsm                      magmablas_ztrsm_q
#define magmablas_ztrsm_outofplace           magmablas_ztrsm_outofplace_q
#define magmablas_ztrsm_work                 magmablas_ztrsm_work_q


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

#undef magma_zsetvector
#undef magma_zgetvector
#undef magma_zcopyvector

#define magma_zsetvector                     magma_zsetvector_q
#define magma_zgetvector                     magma_zgetvector_q
#define magma_zcopyvector                    magma_zcopyvector_q


// ========================================
// copying sub-matrices (contiguous columns)

#undef magma_zsetmatrix
#undef magma_zgetmatrix
#undef magma_zcopymatrix

#define magma_zsetmatrix                     magma_zsetmatrix_q
#define magma_zgetmatrix                     magma_zgetmatrix_q
#define magma_zcopymatrix                    magma_zcopymatrix_q


// ========================================
// Level 1 BLAS (alphabetical order)

#define magma_izamax                         magma_izamax_q
#define magma_izamin                         magma_izamin_q
#define magma_dzasum                         magma_dzasum_q
#define magma_zaxpy                          magma_zaxpy_q           
#define magma_zcopy                          magma_zcopy_q
#define magma_zdotc                          magma_zdotc_q
#define magma_zdotu                          magma_zdotu_q
#define magma_dznrm2                         magma_dznrm2_q
#define magma_zrot                           magma_zrot_q
#define magma_zdrot                          magma_zdrot_q
                        
#ifdef REAL             
#define magma_zrotm                          magma_zrotm_q
#define magma_zrotmg                         magma_zrotmg_q
#endif                  
                        
#define magma_zscal                          magma_zscal_q
#define magma_zdscal                         magma_zdscal_q
#define magma_zswap                          magma_zswap_q


// ========================================
// Level 2 BLAS (alphabetical order)

#define magma_zgemv                          magma_zgemv_q
#define magma_zgerc                          magma_zgerc_q
#define magma_zgeru                          magma_zgeru_q
#define magma_zhemv                          magma_zhemv_q
#define magma_zher                           magma_zher_q
#define magma_zher2                          magma_zher2_q
#define magma_ztrmv                          magma_ztrmv_q
#define magma_ztrsv                          magma_ztrsv_q


// ========================================
// Level 3 BLAS (alphabetical order)

#define magma_zgemm                          magma_zgemm_q
#define magma_zsymm                          magma_zsymm_q
#define magma_zhemm                          magma_zhemm_q
#define magma_zsyr2k                         magma_zsyr2k_q
#define magma_zher2k                         magma_zher2k_q
#define magma_zsyrk                          magma_zsyrk_q
#define magma_zherk                          magma_zherk_q
#define magma_ztrmm                          magma_ztrmm_q
#define magma_ztrsm                          magma_ztrsm_q

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMABLAS_Z_H */
