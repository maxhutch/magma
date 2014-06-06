#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      December 2013
#//////////////////////////////////////////////////////////////////////////////

DIR = magmablas
MAGMA_DIR = ..
include ../Makefile.internal

ZHDR  = commonblas_z.h

# alphabetic order by base name (ignoring precision)
ZSRC := \
	zauxiliary.cu		\
	zcaxpycp.cu		\
	zclaswp.cu		\
	zgeadd.cu		\
	zgeadd_batched.cu	\
	zgemm_reduce.cu		\
	dgemv_MLU.cu		\
	zlacpy.cu		\
	zlacpy_batched.cu	\
	zlaqps2_gpu.cu		\
	zlaqps3_gpu.cu		\
	zlag2c.cu		\
	clag2z.cu		\
	zlange.cu		\
	zlanhe.cu		\
	zlarf.cu		\
	zlarfbx.cu		\
	zlarfx.cu		\
	dlarfg.cu		\
	zlarfg-v2.cu		\
	zlarfgx-v2.cu		\
	zlascl.cu		\
	zlaswp.cu		\
	zlat2c.cu		\
	dznrm2.cu		\
	stream.cu		\
	zswap.cu		\
	zswapblk.cu		\
	zswapdblk.cu		\
	zsymmetrize.cu		\
	zsymmetrize_tiles.cu	\
	ssyr2k.cu		\
	ztranspose.cu		\
	ztranspose-v2.cu	\
	ztranspose_inplace.cu	\
	zgetmatrix_transpose.cu	\
	zsetmatrix_transpose.cu	\
	dtrsm.cu		\
	dtrsm_work.cu		\

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
	\
	chemv_tesla.cu		\
	csymv_tesla.cu		\

# Fermi-specific (CUDA arch 2.x) codes
ZSRC += \
	zgemm_fermi.cu		\
	\
	sgemv_fermi.cu		\
	dgemv_fermi.cu		\
	zgemv_fermi.cu		\
	\
	zhemv_fermi.cu		\
	zsymv_fermi.cu		\
	#cgemv_fermi.cu		\

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

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR) \
	commonblas.h

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.cu=.cu_o) 
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
zgemm_fermi.cu_o: zgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
cgemm_fermi.cu_o: cgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
dgemm_fermi.cu_o: dgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
sgemm_fermi.cu_o: sgemm_fermi_kernels.h gemm_stencil.cu gemm_stencil_defs.h
