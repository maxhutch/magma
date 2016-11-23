/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z_v1_map.h, normal z -> s, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMABLAS_S_V1_MAP_H
#define MAGMABLAS_S_V1_MAP_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#define magmablas_stranspose_inplace        magmablas_stranspose_inplace_v1
#define magmablas_stranspose_inplace   magmablas_stranspose_inplace_v1
#define magmablas_stranspose                magmablas_stranspose_v1
#define magmablas_stranspose           magmablas_stranspose_v1
#define magmablas_sgetmatrix_transpose      magmablas_sgetmatrix_transpose_v1
#define magmablas_ssetmatrix_transpose      magmablas_ssetmatrix_transpose_v1
#define magmablas_sprbt                     magmablas_sprbt_v1
#define magmablas_sprbt_mv                  magmablas_sprbt_mv_v1
#define magmablas_sprbt_mtv                 magmablas_sprbt_mtv_v1
#define magma_sgetmatrix_1D_col_bcyclic     magma_sgetmatrix_1D_col_bcyclic_v1
#define magma_ssetmatrix_1D_col_bcyclic     magma_ssetmatrix_1D_col_bcyclic_v1
#define magma_sgetmatrix_1D_row_bcyclic     magma_sgetmatrix_1D_row_bcyclic_v1
#define magma_ssetmatrix_1D_row_bcyclic     magma_ssetmatrix_1D_row_bcyclic_v1
#define magmablas_sgeadd                    magmablas_sgeadd_v1
#define magmablas_sgeadd2                   magmablas_sgeadd2_v1
#define magmablas_slacpy                    magmablas_slacpy_v1
#define magmablas_slacpy_conj               magmablas_slacpy_conj_v1
#define magmablas_slacpy_sym_in             magmablas_slacpy_sym_in_v1
#define magmablas_slacpy_sym_out            magmablas_slacpy_sym_out_v1
#define magmablas_slange                    magmablas_slange_v1
#define magmablas_slansy                    magmablas_slansy_v1
#define magmablas_slansy                    magmablas_slansy_v1
#define magmablas_slarfg                    magmablas_slarfg_v1
#define magmablas_slascl                    magmablas_slascl_v1
#define magmablas_slascl_2x2                magmablas_slascl_2x2_v1
#define magmablas_slascl2                   magmablas_slascl2_v1
#define magmablas_slascl_diag               magmablas_slascl_diag_v1
#define magmablas_slaset                    magmablas_slaset_v1
#define magmablas_slaset_band               magmablas_slaset_band_v1
#define magmablas_slaswp                    magmablas_slaswp_v1
#define magmablas_slaswp2                   magmablas_slaswp2_v1
#define magmablas_slaswp_sym                magmablas_slaswp_sym_v1
#define magmablas_slaswpx                   magmablas_slaswpx_v1
#define magmablas_ssymmetrize               magmablas_ssymmetrize_v1
#define magmablas_ssymmetrize_tiles         magmablas_ssymmetrize_tiles_v1
#define magmablas_strtri_diag               magmablas_strtri_diag_v1
#define magmablas_snrm2_adjust             magmablas_snrm2_adjust_v1
#define magmablas_snrm2_check              magmablas_snrm2_check_v1
#define magmablas_snrm2_cols               magmablas_snrm2_cols_v1
#define magmablas_snrm2_row_check_adjust   magmablas_snrm2_row_check_adjust_v1
#define magma_slarfb_gpu                    magma_slarfb_gpu_v1
#define magma_slarfb_gpu_gemm               magma_slarfb_gpu_gemm_v1
#define magma_slarfbx_gpu                   magma_slarfbx_gpu_v1
#define magma_slarfg_gpu                    magma_slarfg_gpu_v1
#define magma_slarfgtx_gpu                  magma_slarfgtx_gpu_v1
#define magma_slarfgx_gpu                   magma_slarfgx_gpu_v1
#define magma_slarfx_gpu                    magma_slarfx_gpu_v1
#define magmablas_saxpycp                   magmablas_saxpycp_v1
#define magmablas_sswap                     magmablas_sswap_v1
#define magmablas_sswapblk                  magmablas_sswapblk_v1
#define magmablas_sswapdblk                 magmablas_sswapdblk_v1
#define magmablas_sgemv                     magmablas_sgemv_v1
#define magmablas_sgemv_conj                magmablas_sgemv_conj_v1
#define magmablas_ssymv                     magmablas_ssymv_v1
#define magmablas_ssymv                     magmablas_ssymv_v1
#define magmablas_sgemm                     magmablas_sgemm_v1
#define magmablas_sgemm_reduce              magmablas_sgemm_reduce_v1
#define magmablas_ssymm                     magmablas_ssymm_v1
#define magmablas_ssymm                     magmablas_ssymm_v1
#define magmablas_ssyr2k                    magmablas_ssyr2k_v1
#define magmablas_ssyr2k                    magmablas_ssyr2k_v1
#define magmablas_ssyrk                     magmablas_ssyrk_v1
#define magmablas_ssyrk                     magmablas_ssyrk_v1
#define magmablas_strsm                     magmablas_strsm_v1
#define magmablas_strsm_outofplace          magmablas_strsm_outofplace_v1
#define magmablas_strsm_work                magmablas_strsm_work_v1

#undef magma_ssetvector
#undef magma_sgetvector
#undef magma_scopyvector
#undef magma_ssetmatrix
#undef magma_sgetmatrix
#undef magma_scopymatrix

#define magma_ssetvector                    magma_ssetvector_v1
#define magma_sgetvector                    magma_sgetvector_v1
#define magma_scopyvector                   magma_scopyvector_v1
#define magma_ssetmatrix                    magma_ssetmatrix_v1
#define magma_sgetmatrix                    magma_sgetmatrix_v1
#define magma_scopymatrix                   magma_scopymatrix_v1

#define magma_isamax                        magma_isamax_v1
#define magma_isamin                        magma_isamin_v1
#define magma_sasum                        magma_sasum_v1
#define magma_saxpy                         magma_saxpy_v1
#define magma_scopy                         magma_scopy_v1
#define magma_sdot                         magma_sdot_v1
#define magma_sdot                         magma_sdot_v1
#define magma_snrm2                        magma_snrm2_v1
#define magma_srot                          magma_srot_v1
#define magma_srot                         magma_srot_v1
#define magma_srotm                         magma_srotm_v1
#define magma_srotmg                        magma_srotmg_v1
#define magma_sscal                         magma_sscal_v1
#define magma_sscal                        magma_sscal_v1
#define magma_sswap                         magma_sswap_v1
#define magma_sgemv                         magma_sgemv_v1
#define magma_sger                         magma_sger_v1
#define magma_sger                         magma_sger_v1
#define magma_ssymv                         magma_ssymv_v1
#define magma_ssyr                          magma_ssyr_v1
#define magma_ssyr2                         magma_ssyr2_v1
#define magma_strmv                         magma_strmv_v1
#define magma_strsv                         magma_strsv_v1
#define magma_sgemm                         magma_sgemm_v1
#define magma_ssymm                         magma_ssymm_v1
#define magma_ssymm                         magma_ssymm_v1
#define magma_ssyr2k                        magma_ssyr2k_v1
#define magma_ssyr2k                        magma_ssyr2k_v1
#define magma_ssyrk                         magma_ssyrk_v1
#define magma_ssyrk                         magma_ssyrk_v1
#define magma_strmm                         magma_strmm_v1
#define magma_strsm                         magma_strsm_v1

#endif // MAGMABLAS_S_V1_MAP_H
