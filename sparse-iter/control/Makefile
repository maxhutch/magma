#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/control
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

INC += -I../include

ZSRC := \
    magma_zutil_sparse.cpp  \
	magma_z_free.cpp	\
	magma_z_init.cpp	\
	magma_z_matrixchar.cpp	\
	magma_z_mconverter.cpp	\
	magma_z_transfer.cpp	\
	magma_z_vio.cpp		\
	magma_zgeneratematrix.cpp	\
	matrix_zio.cpp		\
	magma_zsolverinfo.cpp	\
	magma_ztranspose.cpp	\
    magma_zp2p.cpp   \
    magma_zcsrsplit.cpp   \
    magma_zmscale.cpp   \
    magma_zmdiff.cpp  \

SRC := \
	mmio.cpp		\

-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) $(SRC)

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA_SPARSE)

$(LIBMAGMA_SPARSE): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo *.mod ../include/*.mod sizeptr

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)
