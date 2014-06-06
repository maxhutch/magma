#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      December 2013
#//////////////////////////////////////////////////////////////////////////////

DIR = testing
MAGMA_DIR = ..
include ../Makefile.internal

# ----------
# BLAS and auxiliary
ZSRC := \
	testing_z_cublas_v2.cpp	\
	testing_zgemm.cpp	\
	testing_zgemv.cpp	\
	testing_zhemv.cpp	\
	testing_zsymv.cpp	\
	testing_zherk.cpp	\
	testing_zher2k.cpp	\
	testing_ztrmm.cpp	\
	testing_ztrmv.cpp	\
	testing_ztrsm.cpp	\
	testing_ztrsv.cpp	\
	\
	testing_auxiliary.cpp	\
	testing_parse_opts.cpp	\
	testing_zblas.cpp	\
	testing_zgeadd.cpp	\
	testing_zgeadd_batched.cpp	\
	testing_zlacpy.cpp	\
	testing_zlacpy_batched.cpp	\
	testing_zlange.cpp	\
	testing_zlanhe.cpp	\
	testing_dlarfg.cpp	\
	testing_zprint.cpp	\
	testing_zsymmetrize.cpp	\
	testing_zsymmetrize_tiles.cpp	\
	testing_zswap.cpp	\
	testing_ztranspose.cpp	\
	testing_zhemv_mgpu.cpp	\
	\
	testing_zhemm_mgpu.cpp	\
	testing_zher2k_mgpu.cpp	\

# ----------
# Cholesky, GPU interface
ZSRC += \
	testing_zcposv_gpu.cpp	\
	\
	testing_zposv_gpu.cpp	\
	testing_zpotrf_gpu.cpp	\
	testing_zpotf2_gpu.cpp	\
	testing_zpotri_gpu.cpp	\
	\
	testing_zpotrf_mgpu.cpp	\

# Cholesky, CPU interface
ZSRC += \
	testing_zposv.cpp	\
	testing_zpotrf.cpp	\
	testing_zpotri.cpp	\

# ----------
# LU, GPU interface
ZSRC += \
	testing_zcgesv_gpu.cpp	\
	\
	testing_zgesv_gpu.cpp	\
	testing_zgetrf_gpu.cpp	\
	testing_zgetf2_gpu.cpp	\
	testing_zgetri_gpu.cpp	\
	\
	testing_zgetrf_mgpu.cpp	\

# LU, CPU interface
ZSRC += \
	testing_zgesv.cpp	\
	testing_zgetrf.cpp	\

# ----------
# QR and least squares, GPU interface
ZSRC += \
	testing_zcgeqrsv_gpu.cpp	\
	\
	testing_zgelqf_gpu.cpp	\
	testing_zgeqrf_gpu.cpp	\
	testing_zgeqr2_gpu.cpp	\
	testing_zgeqr2x_gpu.cpp	\
	testing_zgels3_gpu.cpp	\
	testing_zgels_gpu.cpp	\
	testing_zlarfb_gpu.cpp	\
	testing_zungqr_gpu.cpp	\
	testing_zunmqr_gpu.cpp	\
	\
	testing_zgeqrf_mgpu.cpp	\
	testing_zgegqr_gpu.cpp  \


# QR, CPU interface
ZSRC += \
	testing_zgelqf.cpp	\
	testing_zgeqlf.cpp	\
	testing_zgeqrf.cpp	\
	testing_zungqr.cpp	\
	testing_zunmqr.cpp	\
	\
	testing_zungqr_m.cpp	\
	\
	testing_zgeqp3.cpp	\
	testing_zgeqp3_gpu.cpp	\

# ----------
# symmetric eigenvalues, GPU interface
ZSRC += \
	testing_dsyevd_gpu.cpp	\
	testing_zheevd_gpu.cpp	\
	testing_zhetrd_gpu.cpp	\

# symmetric eigenvalues, CPU interface
ZSRC += \
	testing_dsyevd.cpp	\
	testing_zheevd.cpp	\
	testing_zhetrd.cpp	\

# symmetric eigenvalues, 2-stage
ZSRC += \
	testing_zhetrd_he2hb.cpp	\
	testing_zheevdx_2stage.cpp	\
#	testing_zhetrd_he2hb_mgpu.cpp	\

# generalized symmetric eigenvalues
ZSRC += \
	testing_dsygvd.cpp	\
	testing_zhegvd.cpp	\
	testing_zhegvd_m.cpp	\
	testing_zhegvdx.cpp	\
	testing_zhegvdx_2stage.cpp	\
	testing_zhegvdx_2stage_m.cpp	\
# ----------
# non-symmetric eigenvalues
ZSRC += \
	testing_dgeev.cpp	\
	testing_zgeev.cpp	\
	testing_zgehrd.cpp	\
	testing_dgeev_m.cpp	\
	testing_zgeev_m.cpp	\
	testing_zgehrd_m.cpp	\

# ----------
# SVD
ZSRC += \
	testing_zgesvd.cpp	\
	testing_zgebrd.cpp	\

# ----------
ifeq ($(FORT), pgfortran)
	FOBJ  = fortran_thunking.o
	ZSRCF = testing_zgetrf_gpu_f.cuf
else
	FOBJ  = fortran.o
	ZSRCF = testing_zgetrf_gpu_f.F90	\
		testing_zgetrf_f.f90
endif

ZSRC += $(ZSRCF)


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)
ALLOBJ := $(ALLOBJ:.cuf=.o)
ALLOBJ := $(ALLOBJ:.cpp=.o)

EXE    := $(ALLOBJ:.o=)

# --------------------
LIBTEST := libtest.a
LIBSRC  := testing_zutil.o testing_cutil.o testing_dutil.o testing_sutil.o testing_util.o
LIBOBJ  := $(LIBSRC:.cpp=.o)
LIBEXE  := $(LIBOBJ:.o=)

# testing_zutil.cpp isn't in ZSRC (since they don't generate EXEs),
# so add generation rules here instead of .Makefile.gen
testing_cutil.cpp: testing_zutil.cpp
	$(PYTHON) ../tools/codegen.py  -p c --file testing_zutil.cpp

testing_dutil.cpp: testing_zutil.cpp
	$(PYTHON) ../tools/codegen.py  -p d --file testing_zutil.cpp

testing_sutil.cpp: testing_zutil.cpp
	$(PYTHON) ../tools/codegen.py  -p s --file testing_zutil.cpp

generate: testing_cutil.cpp testing_dutil.cpp testing_sutil.cpp


# --------------------
.PHONY: all lib clean cleanall cleanexe

.DEFAULT_GOAL := all
all: $(EXE)

lib:
	cd .. && $(MAKE) lib

# depend on header
$(ALLOBJ): testings.h

$(LIBTEST): $(LIBOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

LIBLAPACKTEST = lin/liblapacktest.a

$(LIBLAPACKTEST):
	cd lin && $(MAKE) liblapacktest.a

$(EXE): $(LIBMAGMA) $(LIBLAPACKTEST) $(LIBTEST)

clean:
	cd lin    && $(MAKE) clean
	cd matgen && $(MAKE) clean
	rm -f *.o *~ $(LIBTEST)

cleanexe:
	rm -f $(EXE)

cleanall: clean cleanexe

ifeq ($(FORT), pgfortran)
testing_%.o : testing_%.cuf
	$(FORT) $(FOPTS) $(INC_MAGMA) -c $< -o $@ 
else
testing_%.o : testing_%.f90
	$(FORT) $(FOPTS) $(INC_MAGMA) -c $< -o $@ 
endif

testing_%_f : testing_%_f.o $(FOBJ)
	$(FORT) $(LDOPTS) $< $(FOBJ) -o $@ \
	$(LIBTEST) $(LIBLAPACKTEST) -L../lib -lmagma \
	$(LIBDIR) \
	$(LIB) 

testing_% : testing_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) $(LIBLAPACKTEST) -L../lib -lmagma \
	$(LIBDIR) \
	$(LIB) 

#
# These two files are provided by cuda
# CUDA 5 needs -DCUBLAS_GFORTRAN or appropriate setting to compile cuda/src/fortran.c bindings
#
fortran.o : OPTS += -DCUBLAS_GFORTRAN
fortran.o : $(CUDADIR)/src/fortran.c
	$(CC) $(OPTS) $(INC_MAGMA) -c $< -o $@

fortran_thunking.o : $(CUDADIR)/src/fortran_thunking.c
	$(CC) $(OPTS) $(INC_MAGMA) -c $< -o $@

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
