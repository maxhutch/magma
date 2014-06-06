#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      December 2013
#//////////////////////////////////////////////////////////////////////////////

#MAGMA_DIR = ../../..
include ../../../make.inc
#include Makefile.ig2 

LIBNAME = ../lib/libcheck.a
lib: $(LIBNAME)
ALLOBJ=dcheck_eig.o dscala.o dscalde.o dstt21.o dsbt21.o dsyt21.o dlarfy.o \
    zcheck_eig.o zstt21.o zhbt21.o zhet21.o zlarfy.o checkdiag.o

$(LIBNAME): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@



.SUFFIXES: .c .o
.SUFFIXES: .f .o
.SUFFIXES: .F .o

.c.o:
	$(CC) $(OPTS) $(INC) -c $*.c

.f.o:
	$(FORT) -O3 -m64 -fPIC  $(INC) -c $*.f

.F.o:
	$(FORT) $(FOPTS) $(INC) -c $*.F

#///////////////////////////////////////////////////////////////////////////////////////////////////

clean:
	rm -f *.o $(LIBNAME)

cleanall:
	rm -f *.o 
#///////////////////////////////////////////////////////////////////////////////////////////////////

