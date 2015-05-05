#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date May 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/control
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

INC += -I../include

ZSRC := \
	error.cpp				\
	magma_zdomainoverlap.cpp    \
	magma_zutil_sparse.cpp	\
	magma_zfree.cpp		    \
	magma_zmatrixchar.cpp	\
	magma_zmconverter.cpp	\
	magma_zmgenerator.cpp	\
	magma_zmio.cpp			\
	magma_zsolverinfo.cpp	\
	magma_zcsrsplit.cpp		\
	magma_ziteriluutils.cpp	\
	magma_zmcsrpass.cpp		\
	magma_zmcsrpass_gpu.cpp	\
	magma_zmcsrcompressor.cpp	\
	magma_zmscale.cpp		\
	magma_zmdiff.cpp		\
	magma_zmlumerge.cpp		\
	magma_zmtranspose.cpp	\
	magma_zmtransfer.cpp	\
	magma_zmilustruct.cpp	\
	magma_zsort.cpp		\
	magma_zvinit.cpp		\
	magma_zvio.cpp			\
	magma_zvtranspose.cpp	\
	magma_zvpass.cpp		\
	magma_zvpass_gpu.cpp	\

SRC := \
	mmio.cpp				\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) $(SRC)

ALLOBJ := $(ALLSRC:.cpp=.$(o_ext))
ALLOBJ := $(ALLOBJ:.f90=.$(o_ext))
ALLOBJ := $(ALLOBJ:.F90=.$(o_ext))

.PHONY: all clean cleanall

all: $(LIBMAGMA_SPARSE)

$(LIBMAGMA_SPARSE): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.$(o_ext) *~ *.linkinfo *.mod ../include/*.mod sizeptr

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)
