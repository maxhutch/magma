MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

#######################################################################
#  This is the makefile to create a library of the test matrix
#  generators used in LAPACK.  The files are organized as follows:
#
#     SCATGEN  -- Auxiliary routines called from both REAL and COMPLEX
#     DZATGEN  -- Auxiliary routines called from both DOUBLE PRECISION
#                 and COMPLEX*16
#     SMATGEN  -- Single precision real matrix generation routines
#     CMATGEN  -- Single precision complex matrix generation routines
#     DMATGEN  -- Double precision real matrix generation routines
#     ZMATGEN  -- Double precision complex matrix generation routines
#
#  The library can be set up to include routines for any combination
#  of the four precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make single
#       make single complex
#       make single double complex complex16
#  Alternatively, the command
#       make
#  without any arguments creates a library of all four precisions.
#  The library is called
#       tmglib.a
#  and is created at the LAPACK directory level.
#
#  To remove the object files after the library is created, enter
#       make clean
#  On some systems, you can force the source files to be recompiled by
#  entering (for example)
#       make single FRC=FRC
#
#######################################################################

TMGLIB = tmglib.a
 
SCATGEN = slatm1.$(o_ext) slaran.$(o_ext) slarnd.$(o_ext)

SMATGEN = slatms.$(o_ext) slatme.$(o_ext) slatmr.$(o_ext) slatmt.$(o_ext) \
   slagge.$(o_ext) slagsy.$(o_ext) slakf2.$(o_ext) slarge.$(o_ext) slaror.$(o_ext) slarot.$(o_ext) slatm2.$(o_ext) \
   slatm3.$(o_ext) slatm5.$(o_ext) slatm6.$(o_ext) slatm7.$(o_ext) slahilb.$(o_ext)

CMATGEN = clatms.$(o_ext) clatme.$(o_ext) clatmr.$(o_ext) clatmt.$(o_ext) \
   clagge.$(o_ext) claghe.$(o_ext) clagsy.$(o_ext) clakf2.$(o_ext) clarge.$(o_ext) claror.$(o_ext) clarot.$(o_ext) \
   clatm1.$(o_ext) clarnd.$(o_ext) clatm2.$(o_ext) clatm3.$(o_ext) clatm5.$(o_ext) clatm6.$(o_ext) clahilb.$(o_ext)

DZATGEN = dlatm1.$(o_ext) dlaran.$(o_ext) dlarnd.$(o_ext)

DMATGEN = dlatms.$(o_ext) dlatme.$(o_ext) dlatmr.$(o_ext) dlatmt.$(o_ext) \
   dlagge.$(o_ext) dlagsy.$(o_ext) dlakf2.$(o_ext) dlarge.$(o_ext) dlaror.$(o_ext) dlarot.$(o_ext) dlatm2.$(o_ext) \
   dlatm3.$(o_ext) dlatm5.$(o_ext) dlatm6.$(o_ext) dlatm7.$(o_ext) dlahilb.$(o_ext)

ZMATGEN = zlatms.$(o_ext) zlatme.$(o_ext) zlatmr.$(o_ext) zlatmt.$(o_ext) \
   zlagge.$(o_ext) zlaghe.$(o_ext) zlagsy.$(o_ext) zlakf2.$(o_ext) zlarge.$(o_ext) zlaror.$(o_ext) zlarot.$(o_ext) \
   zlatm1.$(o_ext) zlarnd.$(o_ext) zlatm2.$(o_ext) zlatm3.$(o_ext) zlatm5.$(o_ext) zlatm6.$(o_ext) zlahilb.$(o_ext)

all:	$(TMGLIB)

ALLOBJ=$(SMATGEN) $(CMATGEN) $(SCATGEN) $(DMATGEN) $(ZMATGEN)	\
	$(DZATGEN)

$(TMGLIB): $(SMATGEN) $(CMATGEN) $(SCATGEN) $(DMATGEN)	\
		$(ZMATGEN) $(DZATGEN)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

single: $(SMATGEN) $(SCATGEN)
	$(ARCH) $(ARCHFLAGS) $(TMGLIB) $(SMATGEN) $(SCATGEN)
	$(RANLIB) $(TMGLIB)

complex: $(CMATGEN) $(SCATGEN)
	$(ARCH) $(ARCHFLAGS) $(TMGLIB) $(CMATGEN) $(SCATGEN)
	$(RANLIB) $(TMGLIB)

double: $(DMATGEN) $(DZATGEN)
	$(ARCH) $(ARCHFLAGS) $(TMGLIB) $(DMATGEN) $(DZATGEN)
	$(RANLIB) $(TMGLIB)

complex16: $(ZMATGEN) $(DZATGEN)
	$(ARCH) $(ARCHFLAGS) $(TMGLIB) $(ZMATGEN) $(DZATGEN)
	$(RANLIB) $(TMGLIB)

$(SCATGEN): $(FRC)
$(SMATGEN): $(FRC)
$(CMATGEN): $(FRC)
$(DZATGEN): $(FRC)
$(DMATGEN): $(FRC)
$(ZMATGEN): $(FRC)
 
FRC:
	@FRC=$(FRC)

clean:
	rm -f *.$(o_ext) $(TMGLIB)

#slaran.$(o_ext): slaran.f
#	$(FORT) $(NOOPT) -c $<
#
#dlaran.$(o_ext): dlaran.f
#	$(FORT) $(NOOPT) -c $<
