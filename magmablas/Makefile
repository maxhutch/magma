#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.7.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = magmablas
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal


ZHDR := \
	commonblas_z.h		\
	ztrtri.cuh		\
	zgerbt.h		\
	zpotf2_devicesfunc.cuh		\
	zlarfg_devicesfunc.cuh		\
	ztrsv_template_device.cuh	\

# additional dependencies
strtri_diag.$(o_ext): strtri.cuh
dtrtri_diag.$(o_ext): dtrtri.cuh
ctrtri_diag.$(o_ext): ctrtri.cuh
ztrtri_diag.$(o_ext): ztrtri.cuh

sgerbt.$(o_ext): sgerbt.h
dgerbt.$(o_ext): dgerbt.h
cgerbt.$(o_ext): cgerbt.h
zgerbt.$(o_ext): zgerbt.h

sgerbt_batched.$(o_ext): sgerbt.h
dgerbt_batched.$(o_ext): dgerbt.h
cgerbt_batched.$(o_ext): cgerbt.h
zgerbt_batched.$(o_ext): zgerbt.h

strtri_diag_batched.$(o_ext): strtri.cuh
dtrtri_diag_batched.$(o_ext): dtrtri.cuh
ctrtri_diag_batched.$(o_ext): ctrtri.cuh
ztrtri_diag_batched.$(o_ext): ztrtri.cuh


# alphabetic order by base name (ignoring precision)
ZSRC := \
	zaxpycp.cu		\
	zcaxpycp.cu		\
	zgeadd.cu		\
	zgemm_reduce.cu		\
	zgemv_conjv.cu		\
	zgerbt_kernels.cu	\
	zgerbt.cu		\
	zhemv.cu		\
	zhemv_upper.cu		\
	zlacpy.cu		\
	zlacpy_cnjg.cu		\
	zlacpy_sym_in.cu	\
	zlacpy_sym_out.cu	\
	zlag2c.cu		\
	clag2z.cu		\
	zlange.cu		\
	zlanhe.cu		\
	zlaqps2_gpu.cu		\
	zlaqps3_gpu.cu		\
	zlarf.cu		\
	zlarfbx.cu		\
	zlarfx.cu		\
	zlarfg.cu		\
	zlarfg-v2.cu		\
	zlarfgx-v2.cu		\
	zlascl.cu		\
	zlascl2.cu		\
	zlascl_2x2.cu		\
	zlascl_diag.cu		\
	zlaset.cu		\
	zlaset_band.cu		\
	zlaswp.cu		\
	zlaswp_sym.cu		\
	zclaswp.cu		\
	zlat2c.cu		\
	clat2z.cu		\
	dznrm2.cu		\
	stream.cpp		\
	zswap.cu		\
	zswapblk.cu		\
	zswapdblk.cu		\
	zsymv.cu		\
	zsymv_upper.cu		\
	zsymmetrize.cu		\
	zsymmetrize_tiles.cu	\
	ztranspose.cu		\
	ztranspose_conj.cu		\
	ztranspose_inplace.cu	\
	ztranspose_conj_inplace.cu	\
	zgetmatrix_transpose.cpp	\
	zsetmatrix_transpose.cpp	\
	ztrsm.cu		\
	ztrtri_diag.cu		\
	ztrtri_lower.cu		\
	ztrtri_upper.cu		\
	zlarft_kernels.cu		\
	ztrsv.cu		\

# Tesla-specific (CUDA arch 1.x) codes
ZSRC += \
	dgemm_tesla.cpp		\
	dgemm_tesla_a_0.cu	\
	dgemm_tesla_ab_0.cu	\
	dgemm_tesla_N_N.cu	\
	dgemm_tesla_N_N_special.cu	\
	dgemm_tesla_N_T.cu	\
	dgemm_tesla_T_N.cu	\
	dgemm_tesla_T_T.cu	\
	dgemm_tesla_T_T_special.cu	\
	\
	sgemv_tesla.cu		\
	dgemv_tesla.cu		\

# Fermi-specific (CUDA arch 2.x) codes
ZSRC += \
	zgemm_fermi.cu		\
	zgemv_fermi.cu		\

# multi-GPU
ZSRC += \
	zbcyclic.cpp		\
	zgetmatrix_transpose_mgpu.cpp	\
	zsetmatrix_transpose_mgpu.cpp	\
	zhemv_mgpu.cu			\
	zhemv_mgpu_upper.cu		\
	zhemm_mgpu.cpp			\
	zhemm_mgpu_spec.cpp		\
	zher2k_mgpu.cpp			\
	zher2k_mgpu_spec.cpp		\
	zherk_mgpu.cpp			\

# panels for one-sided factorizations in CUDA
ZSRC += \
	zgetf2.cu		\
	zgeqr2.cpp		\
	zgeqr2x_gpu-v4.cu	\
	zpotf2.cu		\

# batched files, alphabetic order by base name (ignoring precision)
ZSRC += \
	zgeadd_batched.cu		\
	zgemm_batched.cpp		\
	zgerbt_func_batched.cu		\
	zgetf2_kernels.cu		\
	zlaswp_batched.cu		\
	zpotf2_kernels_old.cu		\
	zpotf2_kernels.cu		\
	zset_pointer.cu			\
	ztrsm_batched.cpp		\
	ztrtri_diag_batched.cu		\
	zgeqr2_kernels.cu		\
	zgeqr2_batched.cu		\
	zherk_batched.cpp		\
	zswapdblk_batched.cu	\
	ztrsv_batched.cu		\

SRC += \
	set_pointer.cu			\
	getrf_setup_pivinfo.cu		\
	zgemv_batched_core.cu		\
	cgemv_batched_core.cu		\
	dgemv_batched_core.cu		\
	sgemv_batched_core.cu		\
	zgemm_batched_core.cu		\
	cgemm_batched_core.cu		\
	dgemm_batched_core.cu		\
	sgemm_batched_core.cu		\
	ssyrk_batched_core.cu		\
	dsyrk_batched_core.cu		\
	cherk_batched_core.cu		\
	zherk_batched_core.cu		\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) $(SRC)

ALLOBJ := $(ALLSRC:.cu=.$(o_ext)) 
ALLOBJ := $(ALLOBJ:.cpp=.$(o_ext))

.PHONY: all clean cleanall

.DEFAULT_GOAL := all
all: $(LIBMAGMA) stamp

$(LIBMAGMA): $(ALLHDR) $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.$(o_ext) *~ *.linkinfo stamp

cleanall: clean
	rm -f $(LIBMAGMA)
# additional dependencies
spotf2_kernels.$(o_ext): spotf2_devicesfunc.cuh
dpotf2_kernels.$(o_ext): dpotf2_devicesfunc.cuh
cpotf2_kernels.$(o_ext): cpotf2_devicesfunc.cuh
zpotf2_kernels.$(o_ext): zpotf2_devicesfunc.cuh
spotf2_kernels_var.$(o_ext): spotf2_devicesfunc.cuh
dpotf2_kernels_var.$(o_ext): dpotf2_devicesfunc.cuh
cpotf2_kernels_var.$(o_ext): cpotf2_devicesfunc.cuh
zpotf2_kernels_var.$(o_ext): zpotf2_devicesfunc.cuh

# extra include dependencies
ztrsv_batched.$(o_ext): ztrsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
ctrsv_batched.$(o_ext): ctrsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
dtrsv_batched.$(o_ext): dtrsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
strsv_batched.$(o_ext): strsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh

ztrsv.$(o_ext): ztrsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
ctrsv.$(o_ext): ctrsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
dtrsv.$(o_ext): dtrsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
strsv.$(o_ext): strsv_template_device.cuh gemv_template_device.cuh gemm_template_device_defs.cuh

zgemv_fermi.$(o_ext): gemv_template_device.cuh gemm_template_device_defs.cuh
cgemv_fermi.$(o_ext): gemv_template_device.cuh gemm_template_device_defs.cuh
dgemv_fermi.$(o_ext): gemv_template_device.cuh gemm_template_device_defs.cuh
sgemv_fermi.$(o_ext): gemv_template_device.cuh gemm_template_device_defs.cuh

zgemv_batched_core.$(o_ext): gemv_template_kernel_batched.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
cgemv_batched_core.$(o_ext): gemv_template_kernel_batched.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
dgemv_batched_core.$(o_ext): gemv_template_kernel_batched.cuh gemv_template_device.cuh gemm_template_device_defs.cuh
sgemv_batched_core.$(o_ext): gemv_template_kernel_batched.cuh gemv_template_device.cuh gemm_template_device_defs.cuh

zgemm_batched_core.$(o_ext): gemm_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
cgemm_batched_core.$(o_ext): gemm_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
dgemm_batched_core.$(o_ext): gemm_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
sgemm_batched_core.$(o_ext): gemm_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh

zherk_batched_core.$(o_ext): herk_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
cherk_batched_core.$(o_ext): herk_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
dsyrk_batched_core.$(o_ext): herk_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
ssyrk_batched_core.$(o_ext): herk_template_kernel_batched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh

sgemm_vbatched.$(o_ext): gemm_template_kernel_vbatched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
dgemm_vbatched.$(o_ext): gemm_template_kernel_vbatched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
cgemm_vbatched.$(o_ext): gemm_template_kernel_vbatched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh
zgemm_vbatched.$(o_ext): gemm_template_kernel_vbatched.cuh gemm_template_device.cuh gemm_template_device_defs.cuh

zgemm_fermi.$(o_ext): zgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
cgemm_fermi.$(o_ext): cgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
dgemm_fermi.$(o_ext): dgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
sgemm_fermi.$(o_ext): sgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
