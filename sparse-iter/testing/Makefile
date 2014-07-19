#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0-beta3) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date July 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/testing
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

LIB += -lcusparse 

# ----------
# utility functions
ZSRC += \
#	testing_z_mv.cpp	\
#	testing_zdot.cpp	\
#	testing_z_multimv.cpp	\
#	testing_z_matrix_io.cpp	\
#    testing_ziluchow.cpp \
#    testing_ziluchowcsrcsc.cpp \
#    testing_ziccchowcsrcsc.cpp \
#    testing_ziluchow_cpu_csrcsc.cpp \
#    testing_ziluchow_multigpu.cpp   \
#    testing_ziluchowcsr_multigpu.cpp    \
#    testing_ziccchow.cpp \
#	testing_zmgenerator.cpp	\
#	testing_z_matrix_compressor.cpp	\

# ----------
# run utility functions
ZSRC += \
#	run_zspmv.cpp	\

# ----------
# test Krylov space linear solvers
ZSRC += \
#	testing_zcg.cpp		\
#	testing_zbicgstab.cpp	\
#	testing_ziterref.cpp		\
#	testing_zgmres.cpp	\
#	testing_zjacobi.cpp	\
#	testing_zp1gmres.cpp	\
#	testing_zbicgstab.cpp	\
#	testing_zir.cpp		\
#	testing_zcir.cpp	\
#	testing_zcpir.cpp	\

# ----------
# run Krylov space linear solvers
ZSRC += \
	run_zcg.cpp		    \
	run_zbicgstab.cpp	\
	run_ziterref.cpp	\
	run_zgmres.cpp	    \
	run_zjacobi.cpp	    \
	run_zbaiter.cpp	    \

# ----------
# run Krylov space eigen-solvers
ZSRC += \
	run_zlobpcg.cpp     \

# ----------
# test Krylov space eigen-solvers
ZSRC += \
#	testing_zlobpcg.cpp     \
#	testing_zlobpcg3.cpp     \

# run preconditioned Krylov solvers
ZSRC += \
	run_zpcg.cpp		    \
	run_zpbicgstab.cpp		    \
	run_zpgmres.cpp		    \

# run solver for laplace problem
ZSRC += \
	run_zlaplacesolver.cpp		    \

# run tester for all solvers on laplace problem
ZSRC += \
	run_zalltests.cpp		    \



# ----------
# test sparse direct solvers
ZSRC += \
#	testing_zbcsrlu.cpp	\

# ----------
# run sparse direct solvers
ZSRC += \
	run_zbcsrlu.cpp	\


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
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB)

run_% : run_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB)

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
