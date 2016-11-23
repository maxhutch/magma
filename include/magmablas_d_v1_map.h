/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z_v1_map.h, normal z -> d, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMABLAS_D_V1_MAP_H
#define MAGMABLAS_D_V1_MAP_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#define magmablas_dtranspose_inplace        magmablas_dtranspose_inplace_v1
#define magmablas_dtranspose_inplace   magmablas_dtranspose_inplace_v1
#define magmablas_dtranspose                magmablas_dtranspose_v1
#define magmablas_dtranspose           magmablas_dtranspose_v1
#define magmablas_dgetmatrix_transpose      magmablas_dgetmatrix_transpose_v1
#define magmablas_dsetmatrix_transpose      magmablas_dsetmatrix_transpose_v1
#define magmablas_dprbt                     magmablas_dprbt_v1
#define magmablas_dprbt_mv                  magmablas_dprbt_mv_v1
#define magmablas_dprbt_mtv                 magmablas_dprbt_mtv_v1
#define magma_dgetmatrix_1D_col_bcyclic     magma_dgetmatrix_1D_col_bcyclic_v1
#define magma_dsetmatrix_1D_col_bcyclic     magma_dsetmatrix_1D_col_bcyclic_v1
#define magma_dgetmatrix_1D_row_bcyclic     magma_dgetmatrix_1D_row_bcyclic_v1
#define magma_dsetmatrix_1D_row_bcyclic     magma_dsetmatrix_1D_row_bcyclic_v1
#define magmablas_dgeadd                    magmablas_dgeadd_v1
#define magmablas_dgeadd2                   magmablas_dgeadd2_v1
#define magmablas_dlacpy                    magmablas_dlacpy_v1
#define magmablas_dlacpy_conj               magmablas_dlacpy_conj_v1
#define magmablas_dlacpy_sym_in             magmablas_dlacpy_sym_in_v1
#define magmablas_dlacpy_sym_out            magmablas_dlacpy_sym_out_v1
#define magmablas_dlange                    magmablas_dlange_v1
#define magmablas_dlansy                    magmablas_dlansy_v1
#define magmablas_dlansy                    magmablas_dlansy_v1
#define magmablas_dlarfg                    magmablas_dlarfg_v1
#define magmablas_dlascl                    magmablas_dlascl_v1
#define magmablas_dlascl_2x2                magmablas_dlascl_2x2_v1
#define magmablas_dlascl2                   magmablas_dlascl2_v1
#define magmablas_dlascl_diag               magmablas_dlascl_diag_v1
#define magmablas_dlaset                    magmablas_dlaset_v1
#define magmablas_dlaset_band               magmablas_dlaset_band_v1
#define magmablas_dlaswp                    magmablas_dlaswp_v1
#define magmablas_dlaswp2                   magmablas_dlaswp2_v1
#define magmablas_dlaswp_sym                magmablas_dlaswp_sym_v1
#define magmablas_dlaswpx                   magmablas_dlaswpx_v1
#define magmablas_dsymmetrize               magmablas_dsymmetrize_v1
#define magmablas_dsymmetrize_tiles         magmablas_dsymmetrize_tiles_v1
#define magmablas_dtrtri_diag               magmablas_dtrtri_diag_v1
#define magmablas_dnrm2_adjust             magmablas_dnrm2_adjust_v1
#define magmablas_dnrm2_check              magmablas_dnrm2_check_v1
#define magmablas_dnrm2_cols               magmablas_dnrm2_cols_v1
#define magmablas_dnrm2_row_check_adjust   magmablas_dnrm2_row_check_adjust_v1
#define magma_dlarfb_gpu                    magma_dlarfb_gpu_v1
#define magma_dlarfb_gpu_gemm               magma_dlarfb_gpu_gemm_v1
#define magma_dlarfbx_gpu                   magma_dlarfbx_gpu_v1
#define magma_dlarfg_gpu                    magma_dlarfg_gpu_v1
#define magma_dlarfgtx_gpu                  magma_dlarfgtx_gpu_v1
#define magma_dlarfgx_gpu                   magma_dlarfgx_gpu_v1
#define magma_dlarfx_gpu                    magma_dlarfx_gpu_v1
#define magmablas_daxpycp                   magmablas_daxpycp_v1
#define magmablas_dswap                     magmablas_dswap_v1
#define magmablas_dswapblk                  magmablas_dswapblk_v1
#define magmablas_dswapdblk                 magmablas_dswapdblk_v1
#define magmablas_dgemv                     magmablas_dgemv_v1
#define magmablas_dgemv_conj                magmablas_dgemv_conj_v1
#define magmablas_dsymv                     magmablas_dsymv_v1
#define magmablas_dsymv                     magmablas_dsymv_v1
#define magmablas_dgemm                     magmablas_dgemm_v1
#define magmablas_dgemm_reduce              magmablas_dgemm_reduce_v1
#define magmablas_dsymm                     magmablas_dsymm_v1
#define magmablas_dsymm                     magmablas_dsymm_v1
#define magmablas_dsyr2k                    magmablas_dsyr2k_v1
#define magmablas_dsyr2k                    magmablas_dsyr2k_v1
#define magmablas_dsyrk                     magmablas_dsyrk_v1
#define magmablas_dsyrk                     magmablas_dsyrk_v1
#define magmablas_dtrsm                     magmablas_dtrsm_v1
#define magmablas_dtrsm_outofplace          magmablas_dtrsm_outofplace_v1
#define magmablas_dtrsm_work                magmablas_dtrsm_work_v1

#undef magma_dsetvector
#undef magma_dgetvector
#undef magma_dcopyvector
#undef magma_dsetmatrix
#undef magma_dgetmatrix
#undef magma_dcopymatrix

#define magma_dsetvector                    magma_dsetvector_v1
#define magma_dgetvector                    magma_dgetvector_v1
#define magma_dcopyvector                   magma_dcopyvector_v1
#define magma_dsetmatrix                    magma_dsetmatrix_v1
#define magma_dgetmatrix                    magma_dgetmatrix_v1
#define magma_dcopymatrix                   magma_dcopymatrix_v1

#define magma_idamax                        magma_idamax_v1
#define magma_idamin                        magma_idamin_v1
#define magma_dasum                        magma_dasum_v1
#define magma_daxpy                         magma_daxpy_v1
#define magma_dcopy                         magma_dcopy_v1
#define magma_ddot                         magma_ddot_v1
#define magma_ddot                         magma_ddot_v1
#define magma_dnrm2                        magma_dnrm2_v1
#define magma_drot                          magma_drot_v1
#define magma_drot                         magma_drot_v1
#define magma_drotm                         magma_drotm_v1
#define magma_drotmg                        magma_drotmg_v1
#define magma_dscal                         magma_dscal_v1
#define magma_dscal                        magma_dscal_v1
#define magma_dswap                         magma_dswap_v1
#define magma_dgemv                         magma_dgemv_v1
#define magma_dger                         magma_dger_v1
#define magma_dger                         magma_dger_v1
#define magma_dsymv                         magma_dsymv_v1
#define magma_dsyr                          magma_dsyr_v1
#define magma_dsyr2                         magma_dsyr2_v1
#define magma_dtrmv                         magma_dtrmv_v1
#define magma_dtrsv                         magma_dtrsv_v1
#define magma_dgemm                         magma_dgemm_v1
#define magma_dsymm                         magma_dsymm_v1
#define magma_dsymm                         magma_dsymm_v1
#define magma_dsyr2k                        magma_dsyr2k_v1
#define magma_dsyr2k                        magma_dsyr2k_v1
#define magma_dsyrk                         magma_dsyrk_v1
#define magma_dsyrk                         magma_dsyrk_v1
#define magma_dtrmm                         magma_dtrmm_v1
#define magma_dtrsm                         magma_dtrsm_v1

#endif // MAGMABLAS_D_V1_MAP_H
