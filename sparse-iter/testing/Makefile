#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date November 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/testing
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

LIB += -lcusparse 

# ----------
# utility functions
ZSRC += \
    testing_zmatrix.cpp     \
    testing_zio.cpp     \
    testing_zmcompressor.cpp     \
    testing_zmconverter.cpp     \


# ----------
# low level LA operations
ZSRC += \
    testing_zdot.cpp        \
    testing_zspmv.cpp       \
    testing_zspmm.cpp       \
    testing_zmadd.cpp     \

# ----------
# low level LA operations
ZSRC += \
    testing_zsolver.cpp     \


# ----------
# Mixed-precision solvers
ZSRC += \
#	testing_zpcg.cpp	\
#	testing_zpbicgstab.cpp	\
#	testing_zcpbicgstab.cpp	\
#	testing_zpgmres.cpp	\
#	testing_zgmres_pipe.cpp	\
#	testing_zcpgmres.cpp	\
#	testing_zdot.cpp	\
#	testing_zkernel.cpp	\
#	testing_z_csrmv.cpp	\
#	testing_z_mpk_4.cpp	\
#	testing_z_mpk_5.cpp	\

# debugging
ZSRC += \
#	testing_zdebug.cpp	\
#   testing_spmvELLRT.cpp \
#	testing_zcdebug.cpp	\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

# path for testings.h, flops.h
INC += -I$(MAGMA_DIR)/testing

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.o)
EXE    := $(ALLOBJ:.o=)

LIBTEST := $(MAGMA_DIR)/testing/libtest.a

.PHONY: all lib clean cleanall

# --------------------
all: $(EXE)

echo:
	@echo EXE $(EXE)
	@echo ALLOBJ $(ALLOBJ)
	@echo ALLSRC $(ALLSRC)
	@echo ZSRC $(ZSRC)

lib:
	cd .. && $(MAKE) lib

# depend on header, in case struct magma_opts changes
$(ALLOBJ): $(MAGMA_DIR)/testing/testings.h


$(EXE): $(LIBMAGMA_SPARSE) $(LIBTEST)

clean:
	rm -f *.o *~ 

cleanexe:
	rm -f $(EXE)

cleanall: clean cleanexe

testing_% : testing_%.o
	echo $(LIB)
	$(CXX) $(LDFLAGS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB)

run_% : run_%.o
	$(CXX) $(LDFLAGS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB)

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
