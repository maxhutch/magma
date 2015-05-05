#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date May 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/src
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

INC += -I$(MAGMA_DIR)/control
INC += -g3


# ------------------------------------------------------------
# Loosely, the order is alphabetic in each section, but mixed precision,
# driver, and multi-GPU routines are separated where it made sense.
# Double precision "d" files that generate "s" files must also be listed
# in ZSRC, despite the name.

# Krylov space linear solvers
ZSRC += \
	zcg.cpp				\
	zcg_res.cpp			\
	zcg_merge.cpp		\
	zbicgstab.cpp		\
	zbicgstab_merge.cpp	\
	zbicgstab_merge2.cpp\
	ziterref.cpp		\
	zjacobi.cpp			\
	zbaiter.cpp			\
	zpcg.cpp			\
	zbpcg.cpp			\
	zfgmres.cpp 		\
	zpbicgstab.cpp		\


# Krylov space eigen-solvers
ZSRC += \
	zlobpcg.cpp			\
#	zcgeig.cpp			\
#	zlancsoz.cpp		\

# Mixed-precision solvers
#ZSRC += \
#	zcpbicgstab.cpp		\
#	zcpgmres.cpp		\
#	zciterref.cpp		\
#	zcpir.cpp			\

# QR and least squares, GPU interface
#ZSRC += \
#	zungqr_lapack.cpp	\
#	zungqr_cgs.cpp		\
#	zungqr_mgs.cpp		\
#	zungqr_chol.cpp		\
#	zungqr_iter.cpp		\

# iterative ILU
ZSRC += \
	ziterilu.cpp			\


# orthogonalization schemes    
ZSRC += \
#	zorthomgs.cpp		\

# backward communication for SpMV and Preconditioner
ZSRC += \
	magma_zcustomspmv.cpp		\
	magma_zcustomprecond.cpp	\

# Wrappers to cusparse functions
ZSRC += \
	zcumilu.cpp			\
	magma_zcuspmm.cpp				\
	magma_zcuspaxpy.cpp				\

# Wrappers, tools etc
ZSRC += \
	magma_z_precond_wrapper.cpp		\
	magma_z_solver_wrapper.cpp		\
	zresidual.cpp					\
	zresidualvec.cpp			\

# research functions
ZSRC += \

# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src
#include "magmasparse_common.h"

ALLSRC_F = \

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.$(o_ext)) $(ALLSRC_F:.f=.$(o_ext))

all: $(LIBMAGMA_SPARSE)

$(LIBMAGMA_SPARSE): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.$(o_ext) *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA_SPARSE)
