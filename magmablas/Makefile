#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = magmablas
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal


ZHDR := \
	commonblas_z.h		\
	ztrtri.h		\
	zgerbt.h		\

# additional dependencies
strtri_diag.o: strtri.h
dtrtri_diag.o: dtrtri.h
ctrtri_diag.o: ctrtri.h
ztrtri_diag.o: ztrtri.h

# additional dependencies
sgerbt.o: sgerbt.h
dgerbt.o: dgerbt.h
cgerbt.o: cgerbt.h
zgerbt.o: zgerbt.h

# additional batched dependencies
strtri_diag_batched.o: strtri.h
dtrtri_diag_batched.o: dtrtri.h
ctrtri_diag_batched.o: ctrtri.h
ztrtri_diag_batched.o: ztrtri.h

# additional batched dependencies
sgerbt_batched.o: sgerbt.h
dgerbt_batched.o: dgerbt.h
cgerbt_batched.o: cgerbt.h
zgerbt_batched.o: zgerbt.h


# alphabetic order by base name (ignoring precision)
ZSRC := \
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
	zclaswp.cu		\
	zlat2c.cu		\
	clat2z.cu		\
	dznrm2.cu		\
	stream.cu		\
	zswap.cu		\
	zswapblk.cu		\
	zswapdblk.cu		\
	zsymv.cu		\
	zsymv_upper.cu		\
	zsymmetrize.cu		\
	zsymmetrize_tiles.cu	\
	ztranspose.cu		\
	ztranspose_inplace.cu	\
	zgetmatrix_transpose.cu	\
	zsetmatrix_transpose.cu	\
	ztrsm.cu		\
	ztrtri_diag.cu		\
	ztrtri_lower.cu		\
	ztrtri_upper.cu		\
	zlarft_kernels.cu		\

# Tesla-specific (CUDA arch 1.x) codes
ZSRC += \
	dgemm_tesla.cu		\
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
	zbcyclic.cu			\
	zgetmatrix_transpose_mgpu.cu	\
	zsetmatrix_transpose_mgpu.cu	\
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
	zgeqr2.cu		\
	zgeqr2x_gpu-v4.cu	\
	zpotf2.cu		\

# batched files, alphabetic order by base name (ignoring precision)
ZSRC += \
	zgeadd_batched.cu		\
	zgemv_batched.cu		\
	zgemm_batched.cu		\
	zgemm_fermi_batched.cu		\
	zgemm_fermi_batched_k32.cu	\
	zgerbt_func_batched.cu		\
	zgetf2_kernels.cu		\
	zherk_batched.cu		\
	zherk_fermi_batched.cu		\
	zherk_fermi_batched_k32.cu	\
	zlaswp_batched.cu		\
	zpotf2_kernels.cu		\
	zset_pointer.cu			\
	ztrsm_batched.cu		\
	ztrtri_diag_batched.cu		\
	zgeqr2_kernels.cu		\
	zgeqr2_batched.cu		\

SRC += \
	set_pointer.cu			\
	getrf_setup_pivinfo.cu		\


-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) $(SRC)

ALLOBJ := $(ALLSRC:.cu=.o) 
ALLOBJ := $(ALLOBJ:.cpp=.o)

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
	rm -f *.cu_o *.o *~ *.linkinfo stamp

cleanall: clean
	rm -f $(LIBMAGMA)

# extra include dependencies
zgemm_fermi.o: zgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
cgemm_fermi.o: cgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
dgemm_fermi.o: dgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h
sgemm_fermi.o: sgemm_fermi_kernels.h gemm_stencil.cuh gemm_kernel.cuh gemm_stencil_defs.h

zgemm_fermi_batched.o: zgemm_fermi_kernels_batched.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h
cgemm_fermi_batched.o: cgemm_fermi_kernels_batched.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h
dgemm_fermi_batched.o: dgemm_fermi_kernels_batched.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h
sgemm_fermi_batched.o: sgemm_fermi_kernels_batched.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h

zgemm_fermi_batched_k32.o: zgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h
cgemm_fermi_batched_k32.o: cgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h
dgemm_fermi_batched_k32.o: dgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h
sgemm_fermi_batched_k32.o: sgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh gemm_kernel_batched.cuh gemm_stencil_defs.h

zherk_fermi_batched.o: zgemm_fermi_kernels_batched.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
cherk_fermi_batched.o: cgemm_fermi_kernels_batched.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
dsyrk_fermi_batched.o: dgemm_fermi_kernels_batched.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
ssyrk_fermi_batched.o: sgemm_fermi_kernels_batched.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h

zherk_fermi_batched_k32.o: zgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
cherk_fermi_batched_k32.o: cgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
dsyrk_fermi_batched_k32.o: dgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
ssyrk_fermi_batched_k32.o: sgemm_fermi_kernels_batched_k32.h gemm_stencil.cuh herk_kernel_batched.cuh gemm_stencil_defs.h
