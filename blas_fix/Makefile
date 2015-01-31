#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = blas_fix
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

# BLAS single & single-complex functions
SRC := \
	cdotc.f		\
	cdotu.f		\
	sasum.f		\
	scabs1.f	\
	scasum.f	\
	scnrm2.f	\
	sdot.f		\
	sdsdot.f	\
	snrm2.f		\

# LAPACK single & single-complex functions
SRC += \
	cladiv.f	\
	clangb.f	\
	clange.f	\
	clangt.f	\
	clanhb.f	\
	clanhe.f	\
	clanhf.f	\
	clanhp.f	\
	clanhs.f	\
	clanht.f	\
	clansb.f	\
	clansp.f	\
	clansy.f	\
	clantb.f	\
	clantp.f	\
	clantr.f	\
	scsum1.f	\
	sladiv.f	\
	slangb.f	\
	slange.f	\
	slangt.f	\
	slanhs.f	\
	slansb.f	\
	slansf.f	\
	slansp.f	\
	slanst.f	\
	slansy.f	\
	slantb.f	\
	slantp.f	\
	slantr.f	\
	slapy2.f	\
	slapy3.f	\
	slamch.f	\

OBJ := $(addsuffix .o, $(basename $(SRC)))

.PHONY: all clean cleanall

.DEFAULT_GOAL := all

all: $(BLAS_FIX)

$(BLAS_FIX): $(OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

clean:
	-rm -f *.o *.a *~ *.linkinfo stamp

cleanall: clean
