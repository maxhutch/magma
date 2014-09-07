#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2014
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = .
include $(MAGMA_DIR)/Makefile.internal
-include Makefile.local


# print CUDA architectures being compiled
# (if this goes in Makefile.internal, it gets printed for each sub-dir)
ifneq ($(findstring sm10, $(GPU_TARGET)),)
    $(info compile for CUDA arch 1.0 (Tesla))
endif
ifneq ($(findstring sm13, $(GPU_TARGET)),)
    $(info compile for CUDA arch 1.3 (Tesla))
endif
ifneq ($(findstring sm20, $(GPU_TARGET)),)
    $(info compile for CUDA arch 2.x (Fermi))
endif
ifneq ($(findstring sm30, $(GPU_TARGET)),)
    $(info compile for CUDA arch 3.0 (Kepler))
endif
ifneq ($(findstring sm35, $(GPU_TARGET)),)
    $(info compile for CUDA arch 3.5 (Kepler))
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
	@echo ======================================== testing
	( cd testing        && $(MAKE) )

clean:
	( cd magmablas      && $(MAKE) clean )
	( cd src            && $(MAKE) clean )
	( cd control        && $(MAKE) clean )
	( cd interface_cuda && $(MAKE) clean )
	( cd testing        && $(MAKE) clean )
	( cd testing/lin    && $(MAKE) clean )
	#(cd quark          && $(MAKE) clean )
	-rm -f $(LIBMAGMA) $(LIBMAGMA_SO)

cleanall:
	( cd magmablas      && $(MAKE) cleanall )
	( cd src            && $(MAKE) cleanall )
	( cd control        && $(MAKE) cleanall )
	( cd interface_cuda && $(MAKE) cleanall )
	( cd testing        && $(MAKE) cleanall )
	( cd testing/lin    && $(MAKE) cleanall )
	( cd lib            && rm -f *.a *.so )
	#(cd quark          && $(MAKE) cleanall )
	$(MAKE) cleanall2

# cleanall2 is a dummy rule to run cleanmkgen at the *end* of make cleanall, so
# .Makefile.gen files aren't deleted and immediately re-created. see Makefile.gen
cleanall2:
	@echo

# filter out MAGMA-specific options for pkg-config
INSTALL_FLAGS := $(filter-out \
	-DMAGMA_SETAFFINITY -DMAGMA_WITH_ACML -DMAGMA_WITH_MKL -DUSE_FLOCK \
	-DMIN_CUDA_ARCH=100 -DMIN_CUDA_ARCH=200 -DMIN_CUDA_ARCH=300 \
	-fno-strict-aliasing -fPIC -O0 -O1 -O2 -O3 -pedantic -stdc++98 \
	-Wall -Wno-long-long, $(CFLAGS))

INSTALL_LDFLAGS := $(filter-out \
	-fPIC -Wall -Xlinker -zmuldefs, $(LDFLAGS))

install_dirs:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib install_dirs
	# MAGMA
	cp $(MAGMA_DIR)/include/*.h  $(prefix)/include
	cp $(LIBMAGMA)               $(prefix)/lib
	-cp $(LIBMAGMA_SO)           $(prefix)/lib
	-cp $(BLAS_FIX)              $(prefix)/lib
	# QUARK
	# cp $(QUARKDIR)/include/quark.h             $(prefix)/include
	# cp $(QUARKDIR)/include/quark_unpack_args.h $(prefix)/include
	# cp $(QUARKDIR)/include/icl_hash.h          $(prefix)/include
	# cp $(QUARKDIR)/include/icl_list.h          $(prefix)/include
	# cp $(QUARKDIR)/lib/libquark.a              $(prefix)/lib
	# pkgconfig
	cat $(MAGMA_DIR)/lib/pkgconfig/magma.pc.in         | \
	    sed -e s:@INSTALL_PREFIX@:"$(prefix)":         | \
	    sed -e s:@CFLAGS@:"$(INSTALL_FLAGS) $(INC)":    | \
	    sed -e s:@LIBS@:"$(INSTALL_LDFLAGS) $(LIBEXT)": | \
	    sed -e s:@MAGMA_REQUIRED@::                      \
	    > $(prefix)/lib/pkgconfig/magma.pc

# ========================================
# This is a crude manner of creating shared libraries.
# First create objects (with -fPIC) and static .a libraries,
# then assume all objects in these directories go into the shared libraries.
# Better solution would be to use non-recursive make, so make knows all the
# objects in each subdirectory, or use libtool, or put rules for, e.g., the
# control directory in src/Makefile (as done in src/CMakeLists.txt)

fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
             $(findstring -fPIC, $(CXXFLAGS)), \
             $(findstring -fPIC, $(FFLAGS)),   \
             $(findstring -fPIC, $(F90FLAGS)), \
             $(findstring -fPIC, $(NVCCFLAGS)))

LIBMAGMA_SO = $(LIBMAGMA:.a=.so)

ifneq ($(fpic),)

shared: lib
	$(MAKE) $(LIBMAGMA_SO)

$(LIBMAGMA_SO): src/*.o control/*.o interface_cuda/*.o magmablas/*.o
	@echo ======================================== $(LIBMAGMA_SO)
	$(CC) $(LDFLAGS) -shared -o $(LIBMAGMA_SO) $^ \
	$(LIBDIR) \
	$(LIB)

else
shared:
	@echo "Error: 'make shared' requires CFLAGS, CXXFLAGS, FFLAGS, F90FLAGS, and NVCCFLAGS to have -fPIC."
	@echo "Please edit your make.inc file. See make.inc.mkl-shared for an example."
	@echo "After updating make.inc, please 'make clean', then 'make shared', then 'make testing'."
endif
