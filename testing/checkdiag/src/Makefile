#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../../..
include $(MAGMA_DIR)/Makefile.internal

LIBNAME = ../lib/libcheck.a
lib: $(LIBNAME)
ALLOBJ=dcheck_eig.o dscala.o dscalde.o dstt21.o dsbt21.o dsyt21.o dlarfy.o \
    zcheck_eig.o zstt21.o zhbt21.o zhet21.o zlarfy.o checkdiag.o

$(LIBNAME): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@


#///////////////////////////////////////////////////////////////////////////////////////////////////

clean:
	rm -f *.o $(LIBNAME)

cleanall:
	rm -f *.o 
#///////////////////////////////////////////////////////////////////////////////////////////////////

