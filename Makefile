#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
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


.PHONY: all lib libmagma test clean cleanall install shared static

.DEFAULT_GOAL := all
all: lib test

static: libmagma

libmagma:
	@echo ======================================== magmablas
	( cd magmablas      && $(MAKE) )
	@echo ======================================== src
	( cd src            && $(MAKE) )
	@echo ======================================== control
	( cd control        && $(MAKE) )
	@echo ======================================== interface
	( cd interface_cuda && $(MAKE) )
	@echo

libquark:
	@echo ======================================== quark
	( cd quark          && $(MAKE) )
	@echo

lapacktest:
	@echo ======================================== lapacktest
	( cd testing/matgen && $(MAKE) )
	( cd testing/lin    && $(MAKE) )
	@echo

test: lib
	@echo ======================================== testing
	( cd testing        && $(MAKE) )
	@echo

clean:
	( cd magmablas      && $(MAKE) clean )
	( cd src            && $(MAKE) clean )
	( cd control        && $(MAKE) clean )
	( cd interface_cuda && $(MAKE) clean )
	( cd testing        && $(MAKE) clean )
	( cd testing/lin    && $(MAKE) clean )
	( cd sparse-iter    && $(MAKE) clean )
	#(cd quark          && $(MAKE) clean )
	-rm -f $(LIBMAGMA) $(LIBMAGMA_SO)
	@echo

cleanall:
	( cd magmablas      && $(MAKE) cleanall )
	( cd src            && $(MAKE) cleanall )
	( cd control        && $(MAKE) cleanall )
	( cd interface_cuda && $(MAKE) cleanall )
	( cd testing        && $(MAKE) cleanall )
	( cd testing/lin    && $(MAKE) cleanall )
	( cd sparse-iter    && $(MAKE) cleanall )
	( cd lib            && rm -f *.a *.so )
	#(cd quark          && $(MAKE) cleanall )
	$(MAKE) cleanall2
	@echo

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
#
# 'make lib' should do the right thing:
# shared if it detects -fPIC in all the variables, otherwise static.

LIBMAGMA_SO = $(LIBMAGMA:.a=.so)

# see Makefile.internal for $(have_fpic) -- boolean for whether all FLAGS have -fPIC

ifneq ($(have_fpic),)
    # ---------- all flags have -fPIC: compile shared & static
lib: static shared

shared: libmagma
	$(MAKE) $(LIBMAGMA_SO)

# MacOS likes the library's path to be set; see make.inc.macos
ifneq ($(INSTALL_NAME),)
    LDFLAGS += $(INSTALL_NAME)$(notdir $(LIBMAGMA_SO))
endif

$(LIBMAGMA_SO): src/*.o control/*.o interface_cuda/*.o magmablas/*.o
	@echo ======================================== $(LIBMAGMA_SO)
	$(CC) $(LDFLAGS) -shared -o $(LIBMAGMA_SO) $^ \
	$(LIBDIR) \
	$(LIB)
	@echo
else
    # ---------- missing -fPIC: compile static only
lib: static

shared:
	@echo "Error: 'make shared' requires CFLAGS, CXXFLAGS, FFLAGS, F90FLAGS, and NVCCFLAGS to have -fPIC."
	@echo "This is now the default in most example make.inc.* files, except atlas."
	@echo "Please edit your make.inc file and uncomment FPIC."
	@echo "After updating make.inc, please 'make clean && make shared && make testing'."
	@echo "To compile only a static library, use 'make static'."
endif
