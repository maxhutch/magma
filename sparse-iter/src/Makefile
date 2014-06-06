#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.5.0-beta2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date May 2014
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
	zcg.cpp			\
	zcg_merge.cpp		\
	zbicgstab.cpp		\
	zbicgstab_merge.cpp	\
	zbicgstab_merge2.cpp	\
	ziterref.cpp		\
	zgmres.cpp		\
	zjacobi.cpp		\
	zbaiter.cpp 	\
	zpcg.cpp		\
    zpbicgstab.cpp		\
    zpgmres.cpp		\
#	ziterref.cpp		\
#	zcg_merge.cpp		\
#	zgmres.cpp		\
#	zpbicgstab.cpp		\
#	zpgmres.cpp		\
#	zgmres_pipe.cpp		\
#	zilucg.cpp		\
#	zilugmres.cpp		\

# Krylov space eigen-solvers
ZSRC += \
	zlobpcg.cpp		\
	#zcgeig.cpp		\
	#zlancsoz.cpp		\

# Mixed-precision solvers
ZSRC += \
#	zcpbicgstab.cpp		\
#	zcpgmres.cpp		\
#	zciterref.cpp		\
#	zcpir.cpp		\

# QR and least squares, GPU interface
ZSRC += \
	#zungqr_lapack.cpp	\
	#zungqr_cgs.cpp		\
	#zungqr_mgs.cpp		\
	#zungqr_chol.cpp	\
	#zungqr_iter.cpp	\

# ILU and sparse direct
ZSRC += \
	zbcsrlu.cpp		\
    zcuilu.cpp		\
    zbailu.cpp		\
    zpastix.cpp     \
#	zilu.cpp		\


# orthogonalization schemes
ZSRC += \
#	zorthomgs.cpp   \




# Wrappers, tools etc
ZSRC += \
	magma_z_precond_wrapper.cpp \
	zresidual.cpp		\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC_F = \

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.o) $(ALLSRC_F:.f=.o)

all: $(LIBMAGMA_SPARSE)

$(LIBMAGMA_SPARSE): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA_SPARSE)
