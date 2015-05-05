#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date May 2015
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal
-include Makefile.local
-include Makefile.gen

LIB += -lcusparse

.PHONY: all lib libmagma_sparse test clean cleanall install shared static

.DEFAULT_GOAL := all
all: lib test

static: libmagma_sparse

libmagma_sparse:
	@echo ======================================== sparse/blas
	( cd blas           && $(MAKE) )
	@echo ======================================== sparse/src
	( cd src            && $(MAKE) )
	@echo ======================================== sparse/control
	( cd control        && $(MAKE) )
	@echo

test: lib
	@echo ======================================== sparse/testing
	( cd testing        && $(MAKE) )
	@echo

clean:
	( cd include        && $(MAKE) clean )
	( cd blas           && $(MAKE) clean )
	( cd src            && $(MAKE) clean )
	( cd control        && $(MAKE) clean )
	( cd testing        && $(MAKE) clean )
	-rm -f $(LIBMAGMA_SPARSE) $(LIBMAGMA_SPARSE_SO)
	@echo

cleanall:
	( cd include        && $(MAKE) cleanall )
	( cd blas           && $(MAKE) cleanall )
	( cd src            && $(MAKE) cleanall )
	( cd control        && $(MAKE) cleanall )
	( cd testing        && $(MAKE) cleanall )
	$(MAKE) cleanall2
	@echo

# cleanall2 is a dummy rule to run cleanmkgen at the *end* of make cleanall, so
# .Makefile.gen files aren't deleted and immediately re-created. see Makefile.gen
cleanall2:
	@echo

##install_dirs:
##	mkdir -p $(prefix)/sparse-iter
##	mkdir -p $(prefix)/sparse-iter/include

##install_dirs
install: lib
	# MAGMA
	cp $(MAGMA_DIR)/sparse-iter/include/*.h  $(prefix)/include
	cp $(LIBMAGMA_SPARSE)                    $(prefix)/lib
	-cp $(LIBMAGMA_SPARSE_SO)                $(prefix)/lib

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

LIBMAGMA_SPARSE_SO = $(LIBMAGMA_SPARSE:.a=.so)

# see Makefile.internal for $(have_fpic) -- boolean for whether all FLAGS have -fPIC

ifneq ($(have_fpic),)
    # ---------- all flags have -fPIC: compile shared & static
lib: static shared

shared: libmagma_sparse
	$(MAKE) $(LIBMAGMA_SPARSE_SO)

# MacOS likes the library's path to be set; see make.inc.macos
ifneq ($(INSTALL_NAME),)
    LDFLAGS += $(INSTALL_NAME)$(notdir $(LIBMAGMA_SPARSE_SO))
endif

$(LIBMAGMA_SPARSE_SO): src/*.$(o_ext) control/*.$(o_ext) blas/*.$(o_ext)
	@echo ======================================== libmagma_sparse.so
	$(CC) $(LDFLAGS) -shared -o $(LIBMAGMA_SPARSE_SO) $^ \
	-L$(MAGMA_DIR)/lib -lmagma \
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
