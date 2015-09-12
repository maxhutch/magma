#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.7.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/blas
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

ZHDR := 

# alphabetic order by base name (ignoring precision)
ZSRC := \
	magma_z_blaswrapper.cpp	\
	zbajac_csr.cu			\
	zgeaxpy.cu              \
	zgecsrmv.cu				\
	zgeellmv.cu				\
	zgeelltmv.cu			\
	zgeellrtmv.cu			\
	zgesellcmv.cu			\
	zgesellcmmv.cu			\
	zjacobisetup.cu			\
	zlobpcg_shift.cu		\
	zlobpcg_residuals.cu	\
	zlobpcg_maxpy.cu		\
	zmdot.cu				\
	zmergebicgstab.cu		\
	zmergebicgstab2.cu		\
    zmergeidr.cu            \
	zmergecg.cu				\
	zmgecsrmv.cu			\
	zmgeellmv.cu			\
	zmgeelltmv.cu			\
	zmgesellcmmv.cu			\
	zpipelinedgmres.cu		\


# Iterative factorizations
ZSRC += \
	ziterilu_kernels.cu	\
	ziteric.cu	\


# Auxiliary routines
ZSRC += \
	zcompact.cu				\
	magma_zmcsrcompressor_gpu.cu	\
	magma_zdiagcheck.cu		\
	zgecsrreimsplit.cu        \
    zgedensereimsplit.cu        \
	
# Tester routines
ZSRC += \

# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.cu=.$(o_ext)) 
ALLOBJ := $(ALLOBJ:.cpp=.$(o_ext))

.PHONY: all clean cleanall

all: $(LIBMAGMA_SPARSE) 

$(LIBMAGMA_SPARSE): $(ALLHDR) $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.$(o_ext) *~ *.linkinfo stamp

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)
