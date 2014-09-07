#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = magmablas
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

ZHDR := \
	commonblas_z.h		\
	ztrtri.h		\

# additional dependencies
strtri_diag.o: strtri.h
dtrtri_diag.o: dtrtri.h
ctrtri_diag.o: ctrtri.h
ztrtri_diag.o: ztrtri.h

# alphabetic order by base name (ignoring precision)
ZSRC := \
	zcaxpycp.cu		\
	zgeadd.cu		\
	zgeadd_batched.cu	\
	zgemm_reduce.cu		\
	zgemv_conjv.cu          \
	zhemv.cu		\
	zlacpy.cu		\
	zlacpy_batched.cu	\
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
	zhemv_fermi_mgpu_32.cu		\
	zhemv_fermi_mgpu.cu		\
	zhemm_mgpu.cpp			\
	zhemm_mgpu_spec.cpp		\
	zher2k_mgpu.cpp			\
	zher2k_mgpu_spec.cpp		\

# panels for one-sided factorizations in CUDA
ZSRC += \
	zgetf2.cu		\
	zgeqr2.cu		\
	zgeqr2x_gpu-v4.cu	\
	zpotf2.cu		\

-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

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
zgemm_fermi.o: zgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
cgemm_fermi.o: cgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
dgemm_fermi.o: dgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
sgemm_fermi.o: sgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
