#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/blas
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

ZHDR  = 

# alphabetic order by base name (ignoring precision)
ZSRC := \
	clag2z_sparse.cu	\
	magma_z_blaswrapper.cpp	\
	magma_clag2z.cpp	\
	magma_zlag2c.cpp	\
	zbajac_csr.cu		\
	zbcsrswp.cu		\
	zbcsrtrsv.cu		\
	zbcsrcpy.cu		\
	zbcsrlugemm.cu		\
	zbcsrlupivloc.cu	\
	zgecsrmv.cu		\
	zgeellmv.cu		\
	zgeelltmv.cu		\
	zgeellrtmv.cu		\
	zgesellcmv.cu		\
	zgesellcmmv.cu		\
	zjacobisetup.cu		\
	zlag2c_sparse.cu	\
	zlobpcg_shift.cu	\
	zlobpcg_residuals.cu	\
	zlobpcg_maxpy.cu	\
	zmdot.cu		\
	zmergebicgstab.cu	\
	zmergebicgstab2.cu	\
	zmergecg.cu		\
	zmgecsrmv.cu		\
	zmgeellmv.cu		\
	zmgeelltmv.cu		\
	zmgesellcmmv.cu		\
	zpipelinedgmres.cu	\


# Auxiliary routines
ZSRC += \
	zcompact.cu		\


-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.cu=.o) 
ALLOBJ := $(ALLOBJ:.cpp=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA_SPARSE) 

$(LIBMAGMA_SPARSE): $(ALLHDR) $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)
