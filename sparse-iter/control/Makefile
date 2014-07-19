#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0-beta3) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date July 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/control
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

INC += -I../include

ZSRC := \
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
    magma_z_chowutils.cpp   \
    magma_zp2p.cpp   \
    magma_zcsrsplit.cpp   \
    magma_zmscale.cpp   \
    magma_zmsort.cpp   \
    magma_zilustruct.cpp   \
    magma_zmhom.cpp   \
	magma_zmreorder.cpp	\

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

$(LIBMAGMA_SPARSE): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo *.mod ../include/*.mod sizeptr

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)
