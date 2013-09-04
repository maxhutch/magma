#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      August 2013
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC := \
	zpanel_to_q.cpp	\
	zprint.cpp	\

ifeq ($(PGIFORTRAN), 0)
ZSRC += \
	magma_zf77.cpp		\
	magma_param.f90		\
	magma_zfortran.F90	\
	magma.f90
else
ZSRC += \
	magma_zf77pgi.cpp
endif

-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)	\
	auxiliary.cpp		\
	get_nb.cpp		\
	timer.cpp		\
	trace.cpp		\
	strlcpy.cpp		\
	bulge_auxiliary.cpp	\
	affinity.cpp		\
	connection_mgpu.cpp	\
	pthread_barrier.cpp	\
	xerbla.cpp		\
	magma_threadsetting.cpp \

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA)

$(LIBMAGMA): stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@
	cp *.mod ../include

# this forces LIBMAGMA to be archived
stamp: $(ALLOBJ)
	touch $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo *.mod ../include/*.mod sizeptr stamp

cleanall: clean
	rm -f $(LIBMAGMA)

# Extra dependencies for .mod files
magma.o: magma_zfortran.o magma_cfortran.o magma_dfortran.o magma_sfortran.o 
magma_zfortran.o: magma_param.o
magma_cfortran.o: magma_param.o
magma_dfortran.o: magma_param.o
magma_sfortran.o: magma_param.o
