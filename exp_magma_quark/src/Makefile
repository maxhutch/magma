#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      September 2013
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal
include ../make.exp.inc

ZSRC := \

# ----------
# parallel memory set		
ZSRC += dmemset_amc.cpp

# ----------
# LU		
ZSRC += dgetrf_gpu_amc.cpp dgetrf_gpu_amc_v2.cpp\
	dgetrf_mgpu_amc.cpp dgetrf_mgpu_amc_v2.cpp dgetrf_mgpu_amc_v3.cpp

-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
	
ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA_QUARK)

$(LIBMAGMA_QUARK): stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLOBJ)
	touch $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo stamp

cleanall: clean
	rm -f $(LIBMAGMA_QUARK)