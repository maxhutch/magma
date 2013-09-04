#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      August 2013
#//////////////////////////////////////////////////////////////////////////////

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
	zhemv_fermi.cu		\
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
	zswap.cu		\
	zswapblk.cu		\
	zswapdblk.cu		\
	zsymmetrize.cu		\
	zsymmetrize_tiles.cu	\
	zsymv_fermi.cu		\
	csymv_tesla.cu		\
	ssyr2k.cu		\
	ztranspose.cu		\
	ztranspose-v2.cu	\
	ztranspose_inplace.cu	\
	zgetmatrix_transpose.cu	\
	zsetmatrix_transpose.cu	\
	dtrsm.cu		\
	strsm.cu		\
	dtrsm_work.cu		\
	strsm_work.cu		\

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
	zpotf2.cu		\
	zgeqr2.cu		\
	zgeqr2x_gpu-v4.cu	\

# Card-specific versions that need precisions generated must be above in ZSRC,
# so they are all put into Makefile.src in the release. The code is #ifdef'd
# for the correct card.

-include Makefile.local
-include Makefile.src

FERMISRC = \
	sgemv_fermi.cu	\
	dgemv_fermi.cu	\
	cgemv_fermi.cu	\
	zgemv_fermi.cu	\
	zgemvt_fermi.cu	\
	\
	sgemm_fermi.cu	\
	sgemm_fermi80.cu\
	sgemm_fermi64.cu\
	dgemm_fermi.cu	\
	zgemm_fermi.cu	\

TESLASRC = \
	sgemv_tesla.cu	\
	dgemv_tesla.cu	\
	cgemv_tesla.cu	\
	zgemv_tesla.cu	\
	gemv32_tesla.cu	\
	zsymv_tesla.cu	\
	zhemv_tesla.cu	\
	chemv_tesla.cu	\
	\
	sgemm_tesla.cu                            \
	sgemm_tesla_a_0.cu                        \
	sgemm_tesla_ab_0.cu                       \
	sgemm_tesla_N_N_64_16_16_16_4.cu          \
	sgemm_tesla_N_N_64_16_16_16_4_special.cu  \
	sgemm_tesla_N_T_64_16_4_16_4.cu           \
	sgemm_tesla_T_N_32_32_8_8_8.cu            \
	sgemm_tesla_T_T_64_16_16_16_4.cu          \
	\
	dgemm_tesla.cu                            \
	dgemm_tesla_a_0.cu                        \
	dgemm_tesla_ab_0.cu                       \
	dgemm_tesla_N_N_64_16_16_16_4.cu          \
	dgemm_tesla_N_N_64_16_16_16_4_special.cu  \
	dgemm_tesla_N_T_64_16_4_16_4.cu           \
	dgemm_tesla_T_N_32_32_8_8_8.cu            \
	dgemm_tesla_T_T_64_16_16_16_4.cu          \
	dgemm_tesla_T_T_64_16_16_16_4_v2.cu       \

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR) \
	commonblas.h

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) \
	dgemv_MLU.cu	\
	stream.cu	\

ifeq (${GPU_TARGET}, Tesla)
    ALLSRC += $(TESLASRC)
else ifeq (${GPU_TARGET}, Fermi)
    ALLSRC += $(FERMISRC)
else ifeq (${GPU_TARGET}, Kepler)
    ALLSRC += $(FERMISRC)
endif

ALLOBJ := $(ALLSRC:.cu=.cu_o) 
ALLOBJ := $(ALLOBJ:.cpp=.o)

.PHONY: all clean cleanall

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
