#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
QUARKDIR = ../../quark
LIBMAGMABLAS = ../../lib/libmagmablas.a
include ../Makefile.internal

INC += -I$(MAGMA_DIR)/control

ZSRC  = zpotrf_mc.cpp	\
        zgetrf_mc.cpp   \
        zgeqrf_mc.cpp   \
        zgeqrf-v2.cpp   \
        zgeqrf-v3.cpp   \
        zlarfb_gpu.cpp   

-include Makefile.src

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ  = $(ALLSRC:.cpp=.o)

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

clean:
	rm -f *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA)
