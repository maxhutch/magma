#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      August 2013
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = blas_z.cpp	\
	zset_get.cpp	\

-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) \
	alloc.cpp	\
	error.cpp	\
	interface.cpp	\
	set_get.cpp	\

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.o *~ *.linkinfo stamp

cleanall: clean
	rm -f $(LIBMAGMA)
