#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      December 2013
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = .
include ./Makefile.internal
-include Makefile.local


# print CUDA architectures being compiled
# (if this goes in Makefile.internal, it gets printed for each sub-dir)
ifneq ($(findstring Tesla, $(GPU_TARGET)),)
    $(info compile for CUDA arch 1.x (Tesla))
endif
ifneq ($(findstring Fermi, $(GPU_TARGET)),)
    $(info compile for CUDA arch 2.x (Fermi))
endif
ifneq ($(findstring Kepler, $(GPU_TARGET)),)
    $(info compile for CUDA arch 3.x (Kepler))
endif


.PHONY: all lib libmagma test clean cleanall install shared

.DEFAULT_GOAL := all
all: lib test

lib: libmagma

libmagma:
	@echo ======================================== magmablas
	( cd magmablas      && $(MAKE) )
	@echo ======================================== src
	( cd src            && $(MAKE) )
	@echo ======================================== control
	( cd control        && $(MAKE) )
	@echo ======================================== interface
	( cd interface_cuda && $(MAKE) )

libquark:
	@echo ======================================== quark
	( cd quark          && $(MAKE) )

lapacktest:
	@echo ======================================== lapacktest
	( cd testing/matgen && $(MAKE) )
	( cd testing/lin    && $(MAKE) )

test: lib
	@echo ======================================== test
	( cd testing        && $(MAKE) )

clean:
	( cd control        && $(MAKE) clean )
	( cd src            && $(MAKE) clean )
	( cd interface_cuda && $(MAKE) clean )
	( cd testing        && $(MAKE) clean )
	( cd testing/lin    && $(MAKE) clean )
	( cd magmablas      && $(MAKE) clean ) 
#	( cd quark          && $(MAKE) clean )
	-rm -f $(LIBMAGMA) $(LIBMAGMA_SO)

cleanall:
	( cd control        && $(MAKE) cleanall )
	( cd src            && $(MAKE) cleanall )
	( cd interface_cuda && $(MAKE) cleanall )
	( cd testing        && $(MAKE) cleanall )
	( cd testing/lin    && $(MAKE) cleanall )
	( cd magmablas      && $(MAKE) cleanall ) 
	( cd lib            && rm -f *.a )
#	( cd quark          && $(MAKE) cleanall )
	$(MAKE) cleanall2

# cleanall2 is a dummy rule to run cleanmkgen at the *end* of make cleanall, so
# .Makefile.gen files aren't deleted and immediately re-created. see Makefile.gen
cleanall2:
	@echo

dir:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib dir
#       MAGMA
	cp $(MAGMA_DIR)/include/*.h  $(prefix)/include
	cp $(LIBMAGMA)               $(prefix)/lib
	-cp $(LIBMAGMA_SO)           $(prefix)/lib
#       QUARK
#	cp $(QUARKDIR)/include/quark.h             $(prefix)/include
#	cp $(QUARKDIR)/include/quark_unpack_args.h $(prefix)/include
#	cp $(QUARKDIR)/include/icl_hash.h          $(prefix)/include
#	cp $(QUARKDIR)/include/icl_list.h          $(prefix)/include
#	cp $(QUARKDIR)/lib/libquark.a              $(prefix)/lib
#       pkgconfig
	cat $(MAGMA_DIR)/lib/pkgconfig/magma.pc.in | \
	    sed -e s:@INSTALL_PREFIX@:"$(prefix)": | \
	    sed -e s:@INCLUDES@:"$(INC)":          | \
	    sed -e s:@LIBEXT@:"$(LIBEXT)":         | \
	    sed -e s:@MAGMA_REQUIRED@::              \
	    > $(prefix)/lib/pkgconfig/magma.pc

# ========================================
# This is a crude manner of creating shared libraries.
# First create objects (with -fPIC) and static .a libraries,
# then assume all objects in these directories go into the shared libraries.
# Better solution would be to use non-recursive make, so make knows all the
# objects in each subdirectory, or use libtool, or put rules for, e.g., the
# control directory in src/Makefile (as done in src/CMakeLists.txt)
LIBMAGMA_SO     = $(LIBMAGMA:.a=.so)

shared: lib
	$(MAKE) $(LIBMAGMA_SO)

$(LIBMAGMA_SO): src/*.o control/*.o interface_cuda/*.o magmablas/*.cu_o
	@echo ======================================== libmagma.so
	$(CC) $(LDOPTS) -shared -o $(LIBMAGMA_SO) \
	src/*.o control/*.o \
	interface_cuda/*.o magmablas/*.cu_o magmablas/*.o \
	$(LIBDIR) \
	$(LIB)
