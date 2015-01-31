#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = src
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

# ------------------------------------------------------------
# Loosely, the order is alphabetic in each section, but mixed precision,
# driver, and multi-GPU routines are separated where it made sense.
# Double precision "d" files that generate "s" files must also be listed
# in ZSRC, despite the name.

# ----------
# BLAS wrappers
ZSRC := \
	cblas_z.cpp		\

# ----------
# Cholesky, GPU interface
ZSRC += \
	zcposv_gpu.cpp		\
	\
	zposv_gpu.cpp		\
	zpotrf_gpu.cpp		\
	zpotri_gpu.cpp		\
	zpotrs_gpu.cpp		\
	zlauum_gpu.cpp		\
	ztrtri_gpu.cpp		\
	\
	zpotrf_mgpu.cpp		\
	zpotrf_mgpu_right.cpp	\
	zpotrf3_mgpu.cpp	\

# Cholesky, CPU interface
ZSRC += \
	zposv.cpp		\
	zpotrf.cpp		\
	zpotri.cpp		\
	zlauum.cpp		\
	ztrtri.cpp		\
	\
	zpotrf_m.cpp		\

# ----------
# LU, GPU interface
ZSRC += \
	zcgesv_gpu.cpp		\
	zcgetrs_gpu.cpp		\
	\
	zgesv_gpu.cpp		\
	zgesv_nopiv_gpu.cpp	\
	zgetrf_gpu.cpp		\
	zgetrf_nopiv_gpu.cpp	\
	zgetri_gpu.cpp		\
	zgetrs_gpu.cpp		\
	zgetrs_nopiv_gpu.cpp	\
	\
	zgetrf_mgpu.cpp		\
	zgetrf2_mgpu.cpp	\
	zgerbt_gpu.cpp		\

# LU, CPU interface
ZSRC += \
	zgesv.cpp		\
	zgesv_rbt.cpp		\
	zgetrf.cpp		\
	zgetf2_nopiv.cpp	\
	zgetrf_nopiv.cpp	\
	\
	zgetrf_m.cpp		\

# LU, auxiliary for multi-gpu-dynamic
ZSRC += \
	zgetrf_incpiv_gpu.cpp	\
	zgessm_gpu.cpp		\
	zssssm_gpu.cpp		\
	ztstrf_gpu.cpp		\

# ----------
# QR and least squares, GPU interface
ZSRC += \
	zcgeqrsv_gpu.cpp	\
	\
	zgelqf_gpu.cpp		\
	zgels3_gpu.cpp		\
	zgels_gpu.cpp		\
	zgegqr_gpu.cpp		\
	zgeqrf2_gpu.cpp		\
	zgeqrf3_gpu.cpp		\
	zgeqrf_gpu.cpp		\
	zgeqr2x_gpu.cpp		\
	zgeqr2x_gpu-v2.cpp	\
	zgeqr2x_gpu-v3.cpp	\
	zgeqrs3_gpu.cpp		\
	zgeqrs_gpu.cpp		\
	zlarfb_gpu.cpp		\
	zlarfb_gpu_gemm.cpp	\
	zungqr_gpu.cpp		\
	zunmql2_gpu.cpp		\
	zunmqr2_gpu.cpp		\
	zunmqr_gpu.cpp		\
	\
	zgeqrf_mgpu.cpp		\
	\
	zgeqp3_gpu.cpp		\
	zlaqps_gpu.cpp		\

# QR, CPU interface
ZSRC += \
	zgelqf.cpp		\
	zgeqlf.cpp		\
	zgeqrf-v4.cpp		\
	zgeqrf.cpp		\
	zgeqrf_ooc.cpp		\
	zungqr.cpp		\
	zungqr2.cpp		\
	zunmlq.cpp		\
	zunmql.cpp		\
	zunmqr.cpp		\
	\
	zgeqp3.cpp		\
	zlaqps.cpp		\
	\
	zungqr_m.cpp		\
	zunmqr_m.cpp		\

# ----------
# symmetric indefinite factorization, CPU interface
ZSRC += \
	zhesv.cpp		\
	zhetrf.cpp		\
	zhetrf_nopiv.cpp	\
	zhetrf_nopiv_cpu.cpp	\
	zhetrf_nopiv_gpu.cpp	\
	zhetrs_nopiv_gpu.cpp	\
	zhesv_nopiv_gpu.cpp	\
	zlahef_gpu.cpp		\

# ----------
# symmetric eigenvalues, GPU interface
ZSRC += \
	dsyevd_gpu.cpp		\
	dsyevdx_gpu.cpp		\
	zheevd_gpu.cpp		\
	zheevdx_gpu.cpp		\
	zheevr_gpu.cpp		\
	zheevx_gpu.cpp		\
	zhetrd2_gpu.cpp		\
	zhetrd_gpu.cpp		\
	zunmtr_gpu.cpp		\

# symmetric eigenvalues, CPU interface
ZSRC += \
	dsyevd.cpp		\
	dsyevdx.cpp		\
	zheevd.cpp		\
	zheevdx.cpp		\
	zheevr.cpp		\
	zheevx.cpp		\
	\
	dlaex0.cpp		\
	dlaex1.cpp		\
	dlaex3.cpp		\
	dmove_eig.cpp		\
	dstedx.cpp		\
	zhetrd.cpp		\
	zlatrd.cpp		\
	zlatrd2.cpp		\
	zstedx.cpp		\
	zungtr.cpp		\
	zunmtr.cpp		\

# symmetric eigenvalues, multi-GPU 
ZSRC += \
	zhetrd_mgpu.cpp		\
	zlatrd_mgpu.cpp		\
	dsyevd_m.cpp		\
	zheevd_m.cpp		\
	dsyevdx_m.cpp		\
	zheevdx_m.cpp		\
	dlaex0_m.cpp		\
	dlaex1_m.cpp		\
	dlaex3_m.cpp		\
	dstedx_m.cpp		\
	zstedx_m.cpp		\
	zunmtr_m.cpp		\

# symmetric eigenvalues 2-stage
ZSRC += \
	dbulge_aux.cpp		\
	zbulge_applyQ.cpp	\
	zbulge_applyQ_v2.cpp	\
	zbulge_aux.cpp		\
	zbulge_kernel.cpp	\
	zbulge_kernel_v2.cpp	\
	zhetrd_he2hb.cpp	\
	zhetrd_hb2st.cpp	\
	zbulge_back.cpp		\
	zungqr_2stage_gpu.cpp	\
	zunmqr_gpu_2stage.cpp	\
	dsyevdx_2stage.cpp	\
	dsygvdx_2stage.cpp	\
	zheevdx_2stage.cpp	\
	zhegvdx_2stage.cpp	\
	\
	zbulge_back_m.cpp	\
	zbulge_applyQ_v2_m.cpp	\
	dsyevdx_2stage_m.cpp	\
	dsygvdx_2stage_m.cpp	\
	zheevdx_2stage_m.cpp	\
	zhegvdx_2stage_m.cpp	\
	zhetrd_he2hb_mgpu.cpp	\
	zhetrd_he2hb_mgpu_spec.cpp	\

# generalized symmetric eigenvalues
ZSRC += \
	dsygvd.cpp		\
	dsygvdx.cpp		\
	zhegst.cpp		\
	zhegvd.cpp		\
	zhegvdx.cpp		\
	zhegvr.cpp		\
	zhegvx.cpp		\
	zhegst_gpu.cpp		\
	\
	zhegst_m.cpp		\
	dsygvd_m.cpp		\
	zhegvd_m.cpp		\
	dsygvdx_m.cpp		\
	zhegvdx_m.cpp		\
	ztrsm_m.cpp		\

# ----------
# non-symmetric eigenvalues
ZSRC += \
	dgeev.cpp		\
	zgeev.cpp		\
	zgehrd.cpp		\
	zgehrd2.cpp		\
	zlahr2.cpp		\
	zlahru.cpp		\
	dlaln2.cpp		\
	dlaqtrsd.cpp		\
	zlatrsd.cpp		\
	dtrevc3.cpp		\
	dtrevc3_mt.cpp		\
	ztrevc3.cpp		\
	ztrevc3_mt.cpp		\
	zunghr.cpp		\

# use ztrevc3 and ztrevc3_mt instead
	#lapack_dtrevc3.f	\
	#lapack_ztrevc3.f	\

# non-symmetric eigenvalues, multi-GPU
ZSRC += \
	dgeev_m.cpp		\
	zgeev_m.cpp		\
	zgehrd_m.cpp		\
	zlahr2_m.cpp		\
	zlahru_m.cpp		\
	zunghr_m.cpp		\

# ----------
# SVD
ZSRC += \
	dgesdd.cpp		\
	zgesdd.cpp		\
	dgesvd.cpp		\
	zgesvd.cpp		\
	zgebrd.cpp		\
	zlabrd_gpu.cpp		\
	zunmbr.cpp		\

# ----------
# Batched, GPU interface
ZSRC += \
	zgetf2_batched.cpp				\
	zgetf2_nopiv_batched.cpp		\
	zgetrf_panel_batched.cpp		\
	zgetrf_panel_nopiv_batched.cpp	\
	zgetrf_batched.cpp				\
	zgetrf_nopiv_batched.cpp		\
	zgetrs_batched.cpp				\
	zgetrs_nopiv_batched.cpp		\
	zgesv_batched.cpp				\
	zgesv_nopiv_batched.cpp			\
	zgerbt_batched.cpp          	\
	zgesv_rbt_batched.cpp			\
	\
	zgetri_outofplace_batched.cpp				\
	\
	zpotf2_batched.cpp				\
	zpotrf_batched.cpp				\
	zpotrf_panel_batched.cpp		\
	zpotrs_batched.cpp				\
	zposv_batched.cpp				\
    \
	zlarft_batched.cpp				\
	zlarfb_gemm_batched.cpp         \
	zgeqrf_panel_batched.cpp		\
	zgeqrf_batched.cpp				\
	
# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f=.o)

.PHONY: all clean cleanall

.DEFAULT_GOAL := all
all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.o *~ *.linkinfo stamp

cleanall: clean
	
	rm -f $(LIBMAGMA)
