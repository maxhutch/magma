#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2014
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal
-include Makefile.local
-include Makefile.gen

#LIB += -lcusparse

.PHONY: all lib libmagma_sparse test clean cleanall install shared

.DEFAULT_GOAL := all
all: lib test

lib: libmagma_sparse

libmagma_sparse:
	@echo ======================================== sparse/blas
	( cd blas           && $(MAKE) )
	@echo ======================================== sparse/src
	( cd src            && $(MAKE) )
	@echo ======================================== sparse/control
	( cd control        && $(MAKE) )

test: lib
	@echo ======================================== sparse/testing
	( cd testing        && $(MAKE) )

clean:
	( cd include        && $(MAKE) clean )
	( cd blas           && $(MAKE) clean )
	( cd src            && $(MAKE) clean )
	( cd control        && $(MAKE) clean )
	( cd testing        && $(MAKE) clean )
	-rm -f $(LIBMAGMA_SPARSE)

cleanall:
	( cd include        && $(MAKE) cleanall )
	( cd blas           && $(MAKE) cleanall )
	( cd src            && $(MAKE) cleanall )
	( cd control        && $(MAKE) cleanall )
	( cd testing        && $(MAKE) cleanall )
	$(MAKE) cleanall2

# cleanall2 is a dummy rule to run cleanmkgen at the *end* of make cleanall, so
# .Makefile.gen files aren't deleted and immediately re-created. see Makefile.gen
cleanall2:
	@echo

dir:
	mkdir -p $(prefix)/sparse-iter
	mkdir -p $(prefix)/sparse-iter/include

install: lib dir
	# MAGMA
	cp $(MAGMA_DIR)/sparse-iter/include/*.h  $(prefix)/sparse-iter/include

# ========================================
# This is a crude manner of creating shared libraries.
# First create objects (with -fPIC) and static .a libraries,
# then assume all objects in these directories go into the shared libraries.
# Better solution would be to use non-recursive make, so make knows all the
# objects in each subdirectory, or use libtool, or put rules for, e.g., the
# control directory in src/Makefile (as done in src/CMakeLists.txt)
LIBMAGMA_SPARSE_SO = $(LIBMAGMA_SPARSE:.a=.so)

shared: lib
	$(MAKE) $(LIBMAGMA_SPARSE_SO)

$(LIBMAGMA_SPARSE_SO): src/*.o control/*.o blas/*.o
	@echo ======================================== libmagma_sparse.so
	$(CC) $(LDFLAGS) -shared -o $(LIBMAGMA_SPARSE_SO) $^ \
	-L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) \
	$(LIB)
