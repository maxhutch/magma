#######################################################################
#  This makefile creates the test programs for the linear equation
#  routines in LAPACK.  The test files are grouped as follows:
#
#       ALINTST -- Auxiliary test routines
#       SLINTST -- Single precision real test routines
#       CLINTST -- Single precision complex test routines
#       SCLNTST -- Single and Complex routines in common
#       DLINTST -- Double precision real test routines
#       ZLINTST -- Double precision complex test routines
#       DZLNTST -- Double and Double Complex routines in common
#
#  Test programs can be generated for all or some of the four different
#  precisions.  Enter make followed by one or more of the data types
#  desired.  Some examples:
#       make single
#       make single complex
#       make single double complex complex16
#  Alternatively, the command
#       make
#  without any arguments creates all four test programs.
#  The executable files are called
#       xlintims, xlintimd, xlintimc, and xlintimz
#  and are created in the next higher directory level.
#
#  To remove the object files after the executable files have been
#  created, enter
#       make clean
#  On some systems, you can force the source files to be recompiled by
#  entering (for example)
#       make single FRC=FRC
#
#######################################################################

DIR = testing/lin
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal


# ----------
# sources for liblapacktest.a, if Fortran is NOT available -- dummy stub functions.
ZSRC := magma_z_no_fortran.cpp

-include Makefile.src

NO_FORT_SRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

NO_FORT_OBJ := $(NO_FORT_SRC:.cpp=.o)

# ----------
# sources for liblapacktest.a, if Fortran IS available.
SRC =   zbdt01.f  \
        zget22.f  \
        zhet21.f  \
        zhst01.f  \
        zunt01.f  \
        zqpt01.f  \
        zqrt02.f  \
        zlarfy.f  \
        zstt21.f  \
        \
        cbdt01.f  \
        cget22.f  \
        chet21.f  \
        chst01.f  \
        cunt01.f  \
        cqpt01.f  \
        cqrt02.f  \
        clarfy.f  \
        cstt21.f  \
        \
        dbdt01.f  \
        dget22.f  \
        dsyt21.f  \
        dhst01.f  \
        dort01.f  \
        dqpt01.f  \
        dqrt02.f  \
        dlarfy.f  \
        dstt21.f  \
        \
        sbdt01.f  \
        sget22.f  \
        ssyt21.f  \
        shst01.f  \
        sort01.f  \
        sqpt01.f  \
        sqrt02.f  \
        slarfy.f  \
        sstt21.f

OBJ = $(SRC:.f=.o)

ifeq ($(FORT),no_fortran)
liblapacktest.a: $(NO_FORT_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@
else
liblapacktest.a: $(OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@
endif


#######################################################################

ALINTST = \
   aladhd.o alaerh.o alaesm.o alahd.o alareq.o \
   alasum.o alasvm.o chkxer.o icopy.o ilaenv.o xlaenv.o xerbla.o

SCLNTST= slaord.o

DZLNTST= dlaord.o 

SLINTST = schkaa.o \
   schkeq.o schkgb.o schkge.o schkgt.o \
   schklq.o schkpb.o schkpo.o schkps.o schkpp.o \
   schkpt.o schkq3.o schkql.o schkqp.o schkqr.o schkrq.o \
   schktb.o schktp.o schktr.o \
   schktz.o \
   sdrvgt.o sdrvls.o sdrvpb.o \
   sdrvpp.o sdrvpt.o sdrvsp.o \
   serrgt.o serrlq.o serrls.o \
   serrps.o serrql.o serrqp.o serrqr.o \
   serrrq.o serrtr.o serrtz.o \
   sgbt01.o sgbt02.o sgbt05.o sgelqs.o sgeqls.o sgeqrs.o \
   sgerqs.o sget01.o sget02.o \
   sget03.o sget04.o sget06.o sget07.o sgtt01.o sgtt02.o \
   sgtt05.o slaptm.o slarhs.o slatb4.o slatb5.o slattb.o slattp.o \
   slattr.o slavsp.o slavsy.o slqt01.o slqt02.o \
   slqt03.o spbt01.o spbt02.o spbt05.o spot01.o \
   spot02.o spot03.o spot05.o spst01.o sppt01.o \
   sppt02.o sppt03.o sppt05.o sptt01.o sptt02.o \
   sptt05.o sqlt01.o sqlt02.o sqlt03.o sqpt01.o \
   sqrt01.o sqrt02.o sqrt03.o sqrt11.o sqrt12.o \
   sqrt13.o sqrt14.o sqrt15.o sqrt16.o sqrt17.o \
   srqt01.o srqt02.o srqt03.o srzt01.o srzt02.o \
   sspt01.o ssyt01.o \
   stbt02.o stbt03.o stbt05.o stbt06.o stpt01.o \
   stpt02.o stpt03.o stpt05.o stpt06.o strt01.o \
   strt02.o strt03.o strt05.o strt06.o \
   stzt01.o stzt02.o sgennd.o

SLINTST += serrvx.o sdrvge.o serrge.o sdrvgb.o sdrvpo.o \
	serrpo.o

CLINTST = cchkaa.o \
   cchkeq.o cchkgb.o cchkge.o cchkgt.o \
   cchklq.o cchkpb.o \
   cchkpo.o cchkps.o cchkpp.o cchkpt.o cchkq3.o cchkql.o cchkqp.o \
   cchkqr.o cchkrq.o cchktb.o \
   cchktp.o cchktr.o cchktz.o \
   cdrvgt.o cdrvhp.o \
   cdrvls.o cdrvpb.o cdrvpp.o cdrvpt.o \
   cdrvsp.o \
   cerrgt.o cerrlq.o \
   cerrls.o cerrps.o cerrql.o cerrqp.o \
   cerrqr.o cerrrq.o cerrtr.o cerrtz.o \
   cgbt01.o cgbt02.o cgbt05.o cgelqs.o cgeqls.o cgeqrs.o \
   cgerqs.o cget01.o cget02.o \
   cget03.o cget04.o cget07.o cgtt01.o cgtt02.o \
   cgtt05.o chet01.o chpt01.o claipd.o claptm.o clarhs.o clatb4.o clatb5.o \
   clatsp.o clatsy.o clattb.o clattp.o clattr.o \
   clavhe.o clavhp.o clavsp.o clavsy.o clqt01.o \
   clqt02.o clqt03.o cpbt01.o cpbt02.o cpbt05.o \
   cpot01.o cpot02.o cpot03.o cpot05.o cpst01.o \
   cppt01.o cppt02.o cppt03.o cppt05.o cptt01.o \
   cptt02.o cptt05.o cqlt01.o cqlt02.o cqlt03.o \
   cqpt01.o cqrt01.o cqrt02.o cqrt03.o cqrt11.o \
   cqrt12.o cqrt13.o cqrt14.o cqrt15.o cqrt16.o \
   cqrt17.o crqt01.o crqt02.o crqt03.o crzt01.o crzt02.o \
   csbmv.o  cspt01.o \
   cspt02.o cspt03.o csyt01.o csyt02.o csyt03.o \
   ctbt02.o ctbt03.o ctbt05.o ctbt06.o ctpt01.o \
   ctpt02.o ctpt03.o ctpt05.o ctpt06.o ctrt01.o \
   ctrt02.o ctrt03.o ctrt05.o ctrt06.o \
   ctzt01.o ctzt02.o sget06.o cgennd.o

CLINTST += cerrvx.o cdrvge.o cdrvgb.o cerrge.o cdrvpo.o \
	cerrpo.o 

DLINTST = dchkaa.o \
   dchkeq.o dchkgb.o dchkge.o dchkgt.o \
   dchklq.o dchkpb.o dchkpo.o dchkps.o dchkpp.o \
   dchkpt.o dchkq3.o dchkql.o dchkqp.o dchkqr.o dchkrq.o \
   dchktb.o dchktp.o dchktr.o \
   dchktz.o \
   ddrvgt.o ddrvls.o ddrvpb.o \
   ddrvpp.o ddrvpt.o ddrvsp.o \
   derrgt.o derrlq.o derrls.o \
   derrps.o derrql.o derrqp.o derrqr.o \
   derrrq.o derrtr.o derrtz.o \
   dgbt01.o dgbt02.o dgbt05.o dgelqs.o dgeqls.o dgeqrs.o \
   dgerqs.o dget01.o dget02.o \
   dget03.o dget04.o dget06.o dget07.o dgtt01.o dgtt02.o \
   dgtt05.o dlaptm.o dlarhs.o dlatb4.o dlatb5.o dlattb.o dlattp.o \
   dlattr.o dlavsp.o dlavsy.o dlqt01.o dlqt02.o \
   dlqt03.o dpbt01.o dpbt02.o dpbt05.o dpot01.o \
   dpot02.o dpot03.o dpot05.o dpst01.o dppt01.o \
   dppt02.o dppt03.o dppt05.o dptt01.o dptt02.o \
   dptt05.o dqlt01.o dqlt02.o dqlt03.o dqpt01.o \
   dqrt01.o dqrt02.o dqrt03.o dqrt11.o dqrt12.o \
   dqrt13.o dqrt14.o dqrt15.o dqrt16.o dqrt17.o \
   drqt01.o drqt02.o drqt03.o drzt01.o drzt02.o \
   dspt01.o dsyt01.o \
   dtbt02.o dtbt03.o dtbt05.o dtbt06.o dtpt01.o \
   dtpt02.o dtpt03.o dtpt05.o dtpt06.o dtrt01.o \
   dtrt02.o dtrt03.o dtrt05.o dtrt06.o \
   dtzt01.o dtzt02.o dgennd.o

DLINTST += derrvx.o ddrvge.o ddrvgb.o derrge.o ddrvpo.o derrpo.o

ZLINTST = zchkaa.o \
   zchkeq.o zchkgb.o zchkge.o zchkgt.o \
   zchklq.o zchkpb.o \
   zchkpo.o zchkps.o zchkpp.o zchkpt.o zchkq3.o zchkql.o zchkqp.o \
   zchkqr.o zchkrq.o zchktb.o \
   zchktp.o zchktr.o zchktz.o \
   zdrvgt.o zdrvhp.o \
   zdrvls.o zdrvpb.o zdrvpp.o zdrvpt.o \
   zdrvsp.o \
   zerrgt.o zerrlq.o \
   zerrls.o zerrps.o zerrql.o zerrqp.o \
   zerrqr.o zerrrq.o zerrtr.o zerrtz.o \
   zgbt01.o zgbt02.o zgbt05.o zgelqs.o zgeqls.o zgeqrs.o \
   zgerqs.o zget01.o zget02.o \
   zget03.o zget04.o zget07.o zgtt01.o zgtt02.o \
   zgtt05.o zhet01.o zhpt01.o zlaipd.o zlaptm.o zlarhs.o zlatb4.o zlatb5.o \
   zlatsp.o zlatsy.o zlattb.o zlattp.o zlattr.o \
   zlavhe.o zlavhp.o zlavsp.o zlavsy.o zlqt01.o \
   zlqt02.o zlqt03.o zpbt01.o zpbt02.o zpbt05.o \
   zpot01.o zpot02.o zpot03.o zpot05.o zpst01.o \
   zppt01.o zppt02.o zppt03.o zppt05.o zptt01.o \
   zptt02.o zptt05.o zqlt01.o zqlt02.o zqlt03.o \
   zqpt01.o zqrt01.o zqrt02.o zqrt03.o zqrt11.o \
   zqrt12.o zqrt13.o zqrt14.o zqrt15.o zqrt16.o \
   zqrt17.o zrqt01.o zrqt02.o zrqt03.o zrzt01.o zrzt02.o \
   zsbmv.o  zspt01.o \
   zspt02.o zspt03.o zsyt01.o zsyt02.o zsyt03.o \
   ztbt02.o ztbt03.o ztbt05.o ztbt06.o ztpt01.o \
   ztpt02.o ztpt03.o ztpt05.o ztpt06.o ztrt01.o \
   ztrt02.o ztrt03.o ztrt05.o ztrt06.o \
   ztzt01.o ztzt02.o dget06.o zgennd.o

ZLINTST += zerrvx.o zdrvge.o zdrvgb.o zerrge.o zdrvpo.o \
	zerrpo.o 

DSLINTST = dchkab.o \
	ddrvab.o ddrvac.o derrab.o derrac.o dget08.o          \
	alaerh.o alahd.o  aladhd.o alareq.o \
	chkxer.o dlarhs.o dlatb4.o xerbla.o \
	dget02.o dpot06.o

ZCLINTST = zchkab.o \
	zdrvab.o zdrvac.o zerrab.o zerrac.o zget08.o          \
	alaerh.o alahd.o  aladhd.o alareq.o \
	chkxer.o zget02.o zlarhs.o zlatb4.o \
	zsbmv.o  xerbla.o zpot06.o zlaipd.o

SLINTSTRFP = schkrfp.o sdrvrfp.o sdrvrf1.o sdrvrf2.o sdrvrf3.o sdrvrf4.o serrrfp.o \
	slatb4.o slarhs.o sget04.o spot01.o spot03.o spot02.o \
	chkxer.o xerbla.o alaerh.o aladhd.o alahd.o alasvm.o 

DLINTSTRFP = dchkrfp.o ddrvrfp.o ddrvrf1.o ddrvrf2.o ddrvrf3.o ddrvrf4.o derrrfp.o \
	dlatb4.o dlarhs.o dget04.o dpot01.o dpot03.o dpot02.o \
	chkxer.o xerbla.o alaerh.o aladhd.o alahd.o alasvm.o 

CLINTSTRFP = cchkrfp.o cdrvrfp.o cdrvrf1.o cdrvrf2.o cdrvrf3.o cdrvrf4.o cerrrfp.o \
	claipd.o clatb4.o clarhs.o csbmv.o cget04.o cpot01.o cpot03.o cpot02.o \
	chkxer.o xerbla.o alaerh.o aladhd.o alahd.o alasvm.o 

ZLINTSTRFP = zchkrfp.o zdrvrfp.o zdrvrf1.o zdrvrf2.o zdrvrf3.o zdrvrf4.o zerrrfp.o \
	zlatb4.o zlaipd.o zlarhs.o zsbmv.o zget04.o zpot01.o zpot03.o zpot02.o \
	chkxer.o xerbla.o alaerh.o aladhd.o alahd.o alasvm.o 

TMGLIB = ../matgen/tmglib.a

all:  single double complex complex16 liblapacktest.a

single: xlintsts
double: xlintstd
complex: xlintstc
complex16: xlintstz 

$(TMGLIB):
	cd ../matgen && $(MAKE)

xlintsts : $(ALINTST) $(SLINTST) $(SCLNTST) ../fortran.o $(TMGLIB)
	$(FORT) $(ALINTST) $(SCLNTST) $(SLINTST) \
	$(TMGLIB) -o xlintsts \
	../fortran.o \
	$(LDFLAGS) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) $(LIB) 

xlintstc : $(ALINTST) $(CLINTST) $(SCLNTST) ../fortran.o $(TMGLIB)
	$(FORT) $(ALINTST) $(SCLNTST) $(CLINTST) \
	$(TMGLIB) -o xlintstc \
	../fortran.o \
	$(LDFLAGS) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) $(LIB) 
 
xlintstd : $(ALINTST) $(DLINTST) $(DZLNTST) ../fortran.o $(TMGLIB)
	$(FORT) $(ALINTST) $(DLINTST) $(DZLNTST) \
	$(TMGLIB) -o xlintstd \
	../fortran.o \
	$(LDFLAGS) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) $(LIB) 
 
xlintstz : $(ALINTST) $(ZLINTST) $(DZLNTST) ../fortran.o $(TMGLIB)
	$(FORT) $(ALINTST) $(DZLNTST) $(ZLINTST) \
	$(TMGLIB) -o xlintstz \
	../fortran.o \
	$(LDFLAGS) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) $(LIB) 
 
$(ALINTST): $(FRC)
$(SCLNTST): $(FRC)
$(DZLNTST): $(FRC)
$(SLINTST): $(FRC)
$(CLINTST): $(FRC)
$(DLINTST): $(FRC)
$(ZLINTST): $(FRC)

FRC:
	@FRC=$(FRC)
 
clean:
	rm -f *.o *.mod liblapacktest.a *~

cleanall: clean
	rm -f liblapacktest.a xlintsts xlintstd xlintstc xlintstz testing_results.txt

# These are handled by generic %.o: %.f rule in Makefile.internal
#schkaa.o: schkaa.f
#	cp $(MAGMA_DIR)/control/magma.mod . 
#	$(FORT) $(FFLAGS) -c $< -o $@
#
#dchkaa.o: dchkaa.f
#	$(FORT) $(FFLAGS) -c $< -o $@
#
#cchkaa.o: cchkaa.f
#	$(FORT) $(FFLAGS) -c $< -o $@
#
#zchkaa.o: zchkaa.f
#	$(FORT) $(FFLAGS) -c $< -o $@

../fortran.o:
	cd .. && $(MAKE) fortran.o
