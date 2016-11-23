/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z_v1_map.h, normal z -> c, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMABLAS_C_V1_MAP_H
#define MAGMABLAS_C_V1_MAP_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#define magmablas_ctranspose_inplace        magmablas_ctranspose_inplace_v1
#define magmablas_ctranspose_conj_inplace   magmablas_ctranspose_conj_inplace_v1
#define magmablas_ctranspose                magmablas_ctranspose_v1
#define magmablas_ctranspose_conj           magmablas_ctranspose_conj_v1
#define magmablas_cgetmatrix_transpose      magmablas_cgetmatrix_transpose_v1
#define magmablas_csetmatrix_transpose      magmablas_csetmatrix_transpose_v1
#define magmablas_cprbt                     magmablas_cprbt_v1
#define magmablas_cprbt_mv                  magmablas_cprbt_mv_v1
#define magmablas_cprbt_mtv                 magmablas_cprbt_mtv_v1
#define magma_cgetmatrix_1D_col_bcyclic     magma_cgetmatrix_1D_col_bcyclic_v1
#define magma_csetmatrix_1D_col_bcyclic     magma_csetmatrix_1D_col_bcyclic_v1
#define magma_cgetmatrix_1D_row_bcyclic     magma_cgetmatrix_1D_row_bcyclic_v1
#define magma_csetmatrix_1D_row_bcyclic     magma_csetmatrix_1D_row_bcyclic_v1
#define magmablas_cgeadd                    magmablas_cgeadd_v1
#define magmablas_cgeadd2                   magmablas_cgeadd2_v1
#define magmablas_clacpy                    magmablas_clacpy_v1
#define magmablas_clacpy_conj               magmablas_clacpy_conj_v1
#define magmablas_clacpy_sym_in             magmablas_clacpy_sym_in_v1
#define magmablas_clacpy_sym_out            magmablas_clacpy_sym_out_v1
#define magmablas_clange                    magmablas_clange_v1
#define magmablas_clanhe                    magmablas_clanhe_v1
#define magmablas_clansy                    magmablas_clansy_v1
#define magmablas_clarfg                    magmablas_clarfg_v1
#define magmablas_clascl                    magmablas_clascl_v1
#define magmablas_clascl_2x2                magmablas_clascl_2x2_v1
#define magmablas_clascl2                   magmablas_clascl2_v1
#define magmablas_clascl_diag               magmablas_clascl_diag_v1
#define magmablas_claset                    magmablas_claset_v1
#define magmablas_claset_band               magmablas_claset_band_v1
#define magmablas_claswp                    magmablas_claswp_v1
#define magmablas_claswp2                   magmablas_claswp2_v1
#define magmablas_claswp_sym                magmablas_claswp_sym_v1
#define magmablas_claswpx                   magmablas_claswpx_v1
#define magmablas_csymmetrize               magmablas_csymmetrize_v1
#define magmablas_csymmetrize_tiles         magmablas_csymmetrize_tiles_v1
#define magmablas_ctrtri_diag               magmablas_ctrtri_diag_v1
#define magmablas_scnrm2_adjust             magmablas_scnrm2_adjust_v1
#define magmablas_scnrm2_check              magmablas_scnrm2_check_v1
#define magmablas_scnrm2_cols               magmablas_scnrm2_cols_v1
#define magmablas_scnrm2_row_check_adjust   magmablas_scnrm2_row_check_adjust_v1
#define magma_clarfb_gpu                    magma_clarfb_gpu_v1
#define magma_clarfb_gpu_gemm               magma_clarfb_gpu_gemm_v1
#define magma_clarfbx_gpu                   magma_clarfbx_gpu_v1
#define magma_clarfg_gpu                    magma_clarfg_gpu_v1
#define magma_clarfgtx_gpu                  magma_clarfgtx_gpu_v1
#define magma_clarfgx_gpu                   magma_clarfgx_gpu_v1
#define magma_clarfx_gpu                    magma_clarfx_gpu_v1
#define magmablas_caxpycp                   magmablas_caxpycp_v1
#define magmablas_cswap                     magmablas_cswap_v1
#define magmablas_cswapblk                  magmablas_cswapblk_v1
#define magmablas_cswapdblk                 magmablas_cswapdblk_v1
#define magmablas_cgemv                     magmablas_cgemv_v1
#define magmablas_cgemv_conj                magmablas_cgemv_conj_v1
#define magmablas_chemv                     magmablas_chemv_v1
#define magmablas_csymv                     magmablas_csymv_v1
#define magmablas_cgemm                     magmablas_cgemm_v1
#define magmablas_cgemm_reduce              magmablas_cgemm_reduce_v1
#define magmablas_chemm                     magmablas_chemm_v1
#define magmablas_csymm                     magmablas_csymm_v1
#define magmablas_csyr2k                    magmablas_csyr2k_v1
#define magmablas_cher2k                    magmablas_cher2k_v1
#define magmablas_csyrk                     magmablas_csyrk_v1
#define magmablas_cherk                     magmablas_cherk_v1
#define magmablas_ctrsm                     magmablas_ctrsm_v1
#define magmablas_ctrsm_outofplace          magmablas_ctrsm_outofplace_v1
#define magmablas_ctrsm_work                magmablas_ctrsm_work_v1

#undef magma_csetvector
#undef magma_cgetvector
#undef magma_ccopyvector
#undef magma_csetmatrix
#undef magma_cgetmatrix
#undef magma_ccopymatrix

#define magma_csetvector                    magma_csetvector_v1
#define magma_cgetvector                    magma_cgetvector_v1
#define magma_ccopyvector                   magma_ccopyvector_v1
#define magma_csetmatrix                    magma_csetmatrix_v1
#define magma_cgetmatrix                    magma_cgetmatrix_v1
#define magma_ccopymatrix                   magma_ccopymatrix_v1

#define magma_icamax                        magma_icamax_v1
#define magma_icamin                        magma_icamin_v1
#define magma_scasum                        magma_scasum_v1
#define magma_caxpy                         magma_caxpy_v1
#define magma_ccopy                         magma_ccopy_v1
#define magma_cdotc                         magma_cdotc_v1
#define magma_cdotu                         magma_cdotu_v1
#define magma_scnrm2                        magma_scnrm2_v1
#define magma_crot                          magma_crot_v1
#define magma_csrot                         magma_csrot_v1
#define magma_crotm                         magma_crotm_v1
#define magma_crotmg                        magma_crotmg_v1
#define magma_cscal                         magma_cscal_v1
#define magma_csscal                        magma_csscal_v1
#define magma_cswap                         magma_cswap_v1
#define magma_cgemv                         magma_cgemv_v1
#define magma_cgerc                         magma_cgerc_v1
#define magma_cgeru                         magma_cgeru_v1
#define magma_chemv                         magma_chemv_v1
#define magma_cher                          magma_cher_v1
#define magma_cher2                         magma_cher2_v1
#define magma_ctrmv                         magma_ctrmv_v1
#define magma_ctrsv                         magma_ctrsv_v1
#define magma_cgemm                         magma_cgemm_v1
#define magma_csymm                         magma_csymm_v1
#define magma_chemm                         magma_chemm_v1
#define magma_csyr2k                        magma_csyr2k_v1
#define magma_cher2k                        magma_cher2k_v1
#define magma_csyrk                         magma_csyrk_v1
#define magma_cherk                         magma_cherk_v1
#define magma_ctrmm                         magma_ctrmm_v1
#define magma_ctrsm                         magma_ctrsm_v1

#endif // MAGMABLAS_C_V1_MAP_H
