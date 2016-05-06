# ----------------------------------------
# programs
#
# Users should make all changes in make.inc
# It should not be necesary to change anything in here.

include make.inc

# defaults if nothing else is given in make.inc
CC         ?= gcc
CXX        ?= g++
NVCC       ?= nvcc
FORT       ?= no_fortran

ARCH       ?= ar
ARCHFLAGS  ?= cr
RANLIB     ?= ranlib

# shared libraries require -fPIC
#FPIC       = -fPIC

# may want -std=c99 for CFLAGS, -std=c++11 for CXXFLAGS
CFLAGS     ?= -O3 $(FPIC) -DADD_ -Wall -MMD
CXXFLAGS   ?= $(CFLAGS)
NVCCFLAGS  ?= -O3         -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function"
FFLAGS     ?= -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS   ?= -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
LDFLAGS    ?= -O3 $(FPIC)

INC        ?= -I$(CUDADIR)/include

LIBDIR     ?= -L$(CUDADIR)/lib
LIB        ?= -lcudart -lcublas -lcusparse -llapack -lblas

GPU_TARGET ?= Fermi Kepler

# Extension for object files: o for unix, obj for Windows?
o_ext      ?= o

prefix     ?= /usr/local/magma



# ----------------------------------------
# MAGMA-specific programs & flags

ifeq ($(blas_fix),1)
    # prepend -lblas_fix to LIB (it must come before LAPACK library/framework)
    LIB := -L./lib -lblas_fix $(LIB)
endif

LIBEXT     = $(LIBDIR) $(LIB)

# preprocessor flags. See below for MAGMA_INC
CPPFLAGS   = $(INC) $(MAGMA_INC)

CFLAGS    += -DHAVE_CUBLAS
CXXFLAGS  += -DHAVE_CUBLAS

# where testers look for MAGMA libraries
RPATH      = -Wl,-rpath,../lib
RPATH2     = -Wl,-rpath,../../lib

codegen    = python tools/codegen.py


# ----------------------------------------
# NVCC options for the different cards
# First, add smXX for architecture names
ifneq ($(findstring Fermi, $(GPU_TARGET)),)
    GPU_TARGET += sm20
endif
ifneq ($(findstring Kepler, $(GPU_TARGET)),)
    GPU_TARGET += sm30 sm35
endif
ifneq ($(findstring Maxwell, $(GPU_TARGET)),)
    GPU_TARGET += sm50
endif

# Next, add compile options for specific smXX
# sm_xx is binary, compute_xx is PTX for forward compatability
# MIN_ARCH is lowest requested version
#          Use it ONLY in magma_print_environment; elsewhere use __CUDA_ARCH__ or magma_getdevice_arch()
# NV_SM    accumulates sm_xx for all requested versions
# NV_COMP  is compute_xx for highest requested version
#
# See also $(info compile for ...) in Makefile
NV_SM    :=
NV_COMP  :=

ifneq ($(findstring sm10, $(GPU_TARGET)),)
    # sm10 is no longer supported by CUDA 6.x nvcc
    #MIN_ARCH ?= 100
    #NV_SM    += -gencode arch=compute_10,code=sm_10
    #NV_COMP  := -gencode arch=compute_10,code=compute_10
    $(warning CUDA arch 1.x is no longer supported by CUDA >= 6.x and MAGMA >= 2.0)
endif
ifneq ($(findstring sm13, $(GPU_TARGET)),)
    #MIN_ARCH ?= 130
    #NV_SM    += -gencode arch=compute_13,code=sm_13
    #NV_COMP  := -gencode arch=compute_13,code=compute_13
    $(warning CUDA arch 1.x is no longer supported by CUDA >= 6.x and MAGMA >= 2.0)
endif
ifneq ($(findstring sm20, $(GPU_TARGET)),)
    MIN_ARCH ?= 200
    NV_SM    += -gencode arch=compute_20,code=sm_20
    NV_COMP  := -gencode arch=compute_20,code=compute_20
endif
ifneq ($(findstring sm30, $(GPU_TARGET)),)
    MIN_ARCH ?= 300
    NV_SM    += -gencode arch=compute_30,code=sm_30
    NV_COMP  := -gencode arch=compute_30,code=compute_30
endif
ifneq ($(findstring sm35, $(GPU_TARGET)),)
    MIN_ARCH ?= 350
    NV_SM    += -gencode arch=compute_35,code=sm_35
    NV_COMP  := -gencode arch=compute_35,code=compute_35
endif
ifneq ($(findstring sm50, $(GPU_TARGET)),)
    MIN_ARCH ?= 500
    NV_SM    += -gencode arch=compute_50,code=sm_50
    NV_COMP  := -gencode arch=compute_50,code=compute_50
endif
ifeq ($(NV_COMP),)
    $(error GPU_TARGET, currently $(GPU_TARGET), must contain one or more of Fermi, Kepler, Maxwell, or sm{20,30,35,50}. Please edit your make.inc file)
endif
NVCCFLAGS += $(NV_SM) $(NV_COMP)
CFLAGS    += -DMIN_CUDA_ARCH=$(MIN_ARCH)
CXXFLAGS  += -DMIN_CUDA_ARCH=$(MIN_ARCH)


# ----------------------------------------
# Define the pointer size for fortran compilation
PTRFILE = control/sizeptr.c
PTROBJ  = control/sizeptr.$(o_ext)
PTREXEC = control/sizeptr
PTRSIZE = $(shell $(PTREXEC))
PTROPT  = -Dmagma_devptr_t="integer(kind=$(PTRSIZE))"

$(PTREXEC): $(PTROBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<


# ----------------------------------------
# include sub-directories

# variables that sub-directories add to.
# these MUST be := defined, not = defined, for $(cdir) to work.
zhdr                :=
libmagma_src        :=
libmagma_zsrc       :=
libsparse_src       :=
libsparse_zsrc      :=
testing_src         :=
testing_zsrc        :=
sparse_testing_zsrc :=

subdirs := \
	blas_fix        \
	control         \
	include         \
	interface_cuda  \
	src             \
	magmablas       \
	testing         \
	testing/lin     \
	sparse-iter     \
	sparse-iter/blas    \
	sparse-iter/control \
	sparse-iter/include \
	sparse-iter/src     \
	sparse-iter/testing \

Makefiles := $(addsuffix /Makefile.src, $(subdirs))

include $(Makefiles)

-include Makefile.internal
-include Makefile.local
-include Makefile.gen
-include Makefile.src


# ----------------------------------------
# objects

# ----- headers
allhdr := $(shdr) $(dhdr) $(chdr) $(zhdr)


# ----- sources & objects
libmagma_allsrc := \
	$(libmagma_ssrc)	\
	$(libmagma_dsrc)	\
	$(libmagma_csrc)	\
	$(libmagma_zsrc)	\
	$(libmagma_src)		\

libsparse_allsrc := \
	$(libsparse_ssrc)	\
	$(libsparse_dsrc)	\
	$(libsparse_csrc)	\
	$(libsparse_zsrc)	\
	$(libsparse_src)	\

libtest_allsrc := \
	$(libtest_ssrc)		\
	$(libtest_dsrc)		\
	$(libtest_csrc)		\
	$(libtest_zsrc)		\
	$(libtest_src)		\

ifeq ($(FORT),no_fortran)
liblapacktest_allsrc := \
	$(liblapacktest_ssrc)	\
	$(liblapacktest_dsrc)	\
	$(liblapacktest_csrc)	\
	$(liblapacktest_zsrc)
else
liblapacktest_allsrc := \
	$(liblapacktest_src)
endif

testing_allsrc := \
	$(testing_ssrc)		\
	$(testing_dsrc)		\
	$(testing_csrc)		\
	$(testing_zsrc)		\

sparse_testing_allsrc := \
	$(sparse_testing_ssrc)	\
	$(sparse_testing_dsrc)	\
	$(sparse_testing_csrc)	\
	$(sparse_testing_zsrc)	\

ifeq ($(FORT),no_fortran)
libmagma_allsrc := $(filter-out %.f %.f90 %.F90, $(libmagma_allsrc))
testing_allsrc  := $(filter-out %.f %.f90 %.F90, $(testing_allsrc))
endif

libmagma_obj       := $(addsuffix .$(o_ext), $(basename $(libmagma_allsrc)))
libsparse_obj      := $(addsuffix .$(o_ext), $(basename $(libsparse_allsrc)))
libblas_fix_obj    := $(addsuffix .$(o_ext), $(basename $(libblas_fix_src)))
libtest_obj        := $(addsuffix .$(o_ext), $(basename $(libtest_allsrc)))
liblapacktest_obj  := $(addsuffix .$(o_ext), $(basename $(liblapacktest_allsrc)))
testing_obj        := $(addsuffix .$(o_ext), $(basename $(testing_allsrc)))
sparse_testing_obj := $(addsuffix .$(o_ext), $(basename $(sparse_testing_allsrc)))

deps :=
deps += $(addsuffix .d, $(basename $(libmagma_allsrc)))
deps += $(addsuffix .d, $(basename $(libsparse_allsrc)))
deps += $(addsuffix .d, $(basename $(libblas_fix_src)))
deps += $(addsuffix .d, $(basename $(libtest_allsrc)))
deps += $(addsuffix .d, $(basename $(liblapacktest_allsrc)))
deps += $(addsuffix .d, $(basename $(testing_allsrc)))
deps += $(addsuffix .d, $(basename $(sparse_testing_allsrc)))

# headers must exist before compiling objects, but we don't want to require
# re-compiling the whole library for every minor header change,
# so use order-only prerequisite (after "|").
$(libmagma_obj):       | $(allhdr)
$(libsparse_obj):      | $(allhdr)
$(libtest_obj):        | $(allhdr)
$(testing_obj):        | $(allhdr)
$(sparse_testing_obj): | $(allhdr)

# changes to testings.h require re-compiling, e.g., if magma_opts changes
$(testing_obj):        testing/testings.h
$(sparse_testing_obj): testing/testings.h

# this allows "make force=force" to force re-compiling
$(libmagma_obj):       $(force)
$(libsparse_obj):      $(force)
$(libblas_fix_obj):    $(force)
$(libtest_obj):        $(force)
$(liblapacktest_obj):  $(force)
$(testing_obj):        $(force)
$(sparse_testing_obj): $(force)

force: ;


# ----- include paths
MAGMA_INC  = -I./include -I./control

$(libtest_obj):        MAGMA_INC += -I./testing
$(testing_obj):        MAGMA_INC += -I./testing
$(libsparse_obj):      MAGMA_INC += -I./sparse-iter/include -I./sparse-iter/control
$(sparse_testing_obj): MAGMA_INC += -I./sparse-iter/include -I./sparse-iter/control -I./testing


# ----- libraries
libmagma_a      := lib/libmagma.a
libmagma_so     := lib/libmagma.so
libsparse_a     := lib/libmagma_sparse.a
libsparse_so    := lib/libmagma_sparse.so
libblas_fix_a   := lib/libblas_fix.a
libtest_a       := testing/libtest.a
liblapacktest_a := testing/lin/liblapacktest.a

# static libraries
libs_a := \
	$(libmagma_a)		\
	$(libsparse_a)		\
	$(libtest_a)		\
	$(liblapacktest_a)	\
	$(libblas_fix_a)	\

# shared libraries
libs_so := \
	$(libmagma_so)		\
	$(libsparse_so)		\

# add objects to libraries
$(libmagma_a):      $(libmagma_obj)
$(libmagma_so):     $(libmagma_obj)
$(libsparse_a):     $(libsparse_obj)
$(libsparse_so):    $(libsparse_obj)
$(libblas_fix_a):   $(libblas_fix_obj)
$(libtest_a):       $(libtest_obj)
$(liblapacktest_a): $(liblapacktest_obj)

# sparse requires libmagma
$(libsparse_so): | $(libmagma_so)


# ----- testers
testing_c_src := $(filter %.c %.cpp,       $(testing_allsrc))
testing_f_src := $(filter %.f %.f90 %.F90, $(testing_allsrc))
testers       := $(basename $(testing_c_src))
testers_f     := $(basename $(testing_f_src))

sparse_testers := $(basename $(sparse_testing_allsrc))

# depend on static libraries
# see below for libmagma, which is either static or shared
$(testers):        $(libtest_a) $(liblapacktest_a)
$(testers_f):      $(libtest_a) $(liblapacktest_a)
$(sparse_testers): $(libtest_a)  # doesn't use liblapacktest

# ----- blas_fix
# if using blas_fix (e.g., on MacOS), libmagma requires libblas_fix
ifeq ($(blas_fix),1)
    $(libmagma_a):     | $(libblas_fix_a)
    $(libmagma_so):    | $(libblas_fix_a)
    $(testers):        | $(libblas_fix_a)
    $(testers_f):      | $(libblas_fix_a)
    $(sparse_testers): | $(libblas_fix_a)
endif


# ----------------------------------------
# MacOS likes shared library's path to be set; see make.inc.macos

ifneq ($(INSTALL_NAME),)
    $(libmagma_so):  LDFLAGS += $(INSTALL_NAME)$(notdir $(libmagma_so))
    $(libsparse_so): LDFLAGS += $(INSTALL_NAME)$(notdir $(libsparse_so))
endif


# ----------------------------------------
# targets

.PHONY: all lib static shared clean test

.DEFAULT_GOAL := all

all: dense sparse

dense: lib test

sparse: sparse-lib sparse-test

# lib defined below in shared libraries, depending on fPIC

test: testing

testers_f: $(testers_f)

sparse-test: sparse-iter/testing

# cleangen is defined in Makefile.gen
cleanall: clean cleangen

# TODO: should this do all $(subdirs) clean?
clean: lib/clean testing/clean
	-rm -f $(deps)


# ----------------------------------------
# shared libraries

# check whether all FLAGS have -fPIC
have_fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
                  $(findstring -fPIC, $(CXXFLAGS)), \
                  $(findstring -fPIC, $(FFLAGS)),   \
                  $(findstring -fPIC, $(F90FLAGS)), \
                  $(findstring -fPIC, $(NVCCFLAGS)))

ifneq ($(have_fpic),)

# --------------------
# if all flags have -fPIC: compile shared & static
lib: static shared

sparse-lib: sparse-static sparse-shared

shared: $(libmagma_so)

sparse-shared: $(libsparse_so)

# as a shared library, changing libmagma.so does NOT require re-linking testers,
# so use order-only prerequisite (after "|").
$(testers):        | $(libmagma_a) $(libmagma_so)
$(testers_f):      | $(libmagma_a) $(libmagma_so)
$(sparse_testers): | $(libmagma_a) $(libmagma_so) $(libsparse_a) $(libsparse_so)

libs := $(libmagma_a) $(libmagma_so) $(libsparse_a) $(libsparse_so)

else

# --------------------
# else: some flags are missing -fPIC: compile static only
lib: static

sparse-lib: sparse-static

shared:
	@echo "Error: 'make shared' requires CFLAGS, CXXFLAGS, FFLAGS, F90FLAGS, and NVCCFLAGS to have -fPIC."
	@echo "This is now the default in most example make.inc.* files, except atlas."
	@echo "Please edit your make.inc file and uncomment FPIC."
	@echo "After updating make.inc, please 'make clean && make shared && make test'."
	@echo "To compile only a static library, use 'make static'."

sparse-shared: shared

# as a static library, changing libmagma.a does require re-linking testers,
# so use regular prerequisite.
$(testers):        $(libmagma_a)
$(testers_f):      $(libmagma_a)
$(sparse_testers): $(libmagma_a) $(libsparse_a)

libs := $(libmagma_a) $(libsparse_a)

endif

ifeq ($(blas_fix),1)
    libs += $(libblas_fix_a)
endif


# ----------------------------------------
# static libraries

static: $(libmagma_a)

sparse-static: $(libsparse_a)


# ----------------------------------------
# sub-directory targets

control_obj        := $(filter        control/%.o, $(libmagma_obj))
interface_cuda_obj := $(filter interface_cuda/%.o, $(libmagma_obj))
magmablas_obj      := $(filter      magmablas/%.o, $(libmagma_obj))
src_obj            := $(filter            src/%.o, $(libmagma_obj))

sparse_control_obj := $(filter sparse-iter/control/%.o, $(libsparse_obj))
sparse_blas_obj    := $(filter    sparse-iter/blas/%.o, $(libsparse_obj))
sparse_src_obj     := $(filter     sparse-iter/src/%.o, $(libsparse_obj))

# ----------
# sub-directory builds
include:             $(allhdr)

blas_fix:            $(libblas_fix_a)

control:             $(control_obj)

interface_cuda:      $(interface_cuda_obj)

magmablas:           $(magmablas_obj)

src:                 $(src_obj)

testing:             $(testers)

sparse-iter:         sparse

sparse-iter/blas:    $(sparse_blas_obj)

sparse-iter/control: $(sparse_control_obj)

sparse-iter/src:     $(sparse_src_obj)

sparse-iter/testing: $(sparse_testers)

# ----------
# sub-directory clean
include/clean:
	-rm -f $(shdr) $(dhdr) $(chdr)

blas_fix/clean:
	-rm -f $(libblas_fix_a) $(libblas_fix_obj)

control/clean:
	-rm -f $(control_obj) include/*.mod control/*.mod

interface_cuda/clean:
	-rm -f $(interface_cuda_obj)

magmablas/clean:
	-rm -f $(magmablas_obj)

src/clean:
	-rm -f $(src_obj)

testing/clean: testing/lin/clean
	-rm -f $(testers) $(testers_f) $(testing_obj) \
		$(libtest_a) $(libtest_obj)

testing/lin/clean:
	-rm -f $(liblapacktest_a) $(liblapacktest_obj)

# hmm... what should lib/clean do? just the libraries, not objects?
lib/clean: blas_fix/clean sparse-iter/clean
	-rm -f $(libmagma_a) $(libmagma_so) $(libmagma_obj)

sparse-iter/clean: sparse-iter/testing/clean
	-rm -f $(libsparse_a) $(libsparse_so) $(libsparse_obj)

sparse-iter/blas/clean:
	-rm -f $(sparse_blas_obj)

sparse-iter/control/clean:
	-rm -f $(sparse_control_obj)

sparse-iter/src/clean:
	-rm -f $(sparse_src_obj)

sparse-iter/testing/clean:
	-rm -f $(sparse_testers) $(sparse_testing_obj)


# ----------------------------------------
# rules

.DELETE_ON_ERROR:

.SUFFIXES:

%.$(o_ext): %.f
	$(FORT) $(FFLAGS) -c -o $@ $<

%.$(o_ext): %.f90
	$(FORT) $(F90FLAGS) $(CPPFLAGS) -c -o $@ $<
	-mv $(notdir $(basename $@)).mod include/

%.$(o_ext): %.F90 $(PTREXEC)
	$(FORT) $(F90FLAGS) $(CPPFLAGS) $(PTROPT) -c -o $@ $<
	-mv $(notdir $(basename $@)).mod include/

%.$(o_ext): %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.$(o_ext): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

%.$(o_ext): %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.h
	$(CC) -E $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.c
	$(CC) -E $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.cpp
	$(CXX) -E $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.cpp
	$(NVCC) -E $(NVCCFLAGS) $(CPPFLAGS) -c -o $@ $<

$(libs_a):
	@echo "===== static library $@"
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@
	@echo

ifneq ($(have_fpic),)
$(libmagma_so):
	@echo "===== shared library $@"
	$(CXX) $(LDFLAGS) -shared -o $@ \
		$^ \
		$(LIBEXT)
	@echo

# Can't add -Llib -lmagma to LIBEXT, because that would apply to libsparse_so's
# prerequisites, namely libmagma_so. So libmagma and libsparse need different rules.
# See Make section 6.11 Target-specific Variable Values.
$(libsparse_so):
	@echo "===== shared library $@"
	$(CXX) $(LDFLAGS) -shared -o $@ \
		$^ \
		$(LIBEXT) -Llib -lmagma
	@echo
else
# missing -fPIC: "make shared" prints warning
$(libs_so): shared
endif

# link testing_foo from testing_foo.o
$(testers): %: %.$(o_ext)
	$(CXX) $(LDFLAGS) $(RPATH) \
	-o $@ $< \
	-L./lib -lmagma \
	-L./testing -ltest \
	-L./testing/lin -llapacktest \
	$(LIBEXT)

# link Fortran testing_foo from testing_foo.o
$(testers_f): %: %.$(o_ext) testing/fortran.o
	$(FORT) $(LDFLAGS) $(RPATH) \
	-o $@ $< testing/fortran.o \
	-L./testing -ltest \
	-L./testing/lin -llapacktest \
	-L./lib -lmagma \
	$(LIBEXT)

# link sparse testing_foo from testing_foo.o
$(sparse_testers): %: %.$(o_ext)
	$(CXX) $(LDFLAGS) $(RPATH2) \
	-o $@ $< \
	-L./testing -ltest \
	-L./lib -lmagma_sparse -lmagma \
	$(LIBEXT)


# ----------------------------------------
# filter out MAGMA-specific options for pkg-config
INSTALL_FLAGS := $(filter-out \
	-DMAGMA_NOAFFINITY -DMAGMA_SETAFFINITY -DMAGMA_WITH_ACML -DMAGMA_WITH_MKL -DUSE_FLOCK \
	-DMIN_CUDA_ARCH=100 -DMIN_CUDA_ARCH=200 -DMIN_CUDA_ARCH=300 \
	-DHAVE_CUBLAS \
	-fno-strict-aliasing -fPIC -O0 -O1 -O2 -O3 -pedantic -std=c99 -stdc++98 -stdc++11 \
	-Wall -Wshadow -Wno-long-long, $(CFLAGS))

INSTALL_LDFLAGS := $(filter-out -fPIC -Wall, $(LDFLAGS))

install_dirs:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib sparse-lib install_dirs
	# MAGMA
	cp include/*.h              $(prefix)/include
	cp sparse-iter/include/*.h  $(prefix)/include
	cp $(libs)                  $(prefix)/lib
	# pkgconfig
	cat lib/pkgconfig/magma.pc.in                   | \
	sed -e s:@INSTALL_PREFIX@:"$(prefix)":          | \
	sed -e s:@CFLAGS@:"$(INSTALL_FLAGS) $(INC)":    | \
	sed -e s:@LIBS@:"$(INSTALL_LDFLAGS) $(LIBEXT)": | \
	sed -e s:@MAGMA_REQUIRED@::                       \
	    > $(prefix)/lib/pkgconfig/magma.pc


# ----------------------------------------
# files.txt is nearly all (active) files in SVN, excluding directories. Useful for rsync, etc.
# files-doxygen.txt is all (active) source files in SVN, used by Doxyfile-fast

# excludes non-active directories like obsolete.
# excludes directories by matching *.* files (\w\.\w) and some exceptions like Makefile.
files.txt: force
	svn st -vq \
		| egrep -v '^D|> moved' \
		| perl -pi -e 's/^.{13} +\S+ +\S+ +\S+ +//' | sort \
		| egrep -v '^\.$$|obsolete|deprecated|contrib\b|^exp' \
		| egrep '\w\.\w|Makefile|docs|run' \
		> files.txt
	egrep -v '(\.css|\.f|\.in|\.m|\.mtx|\.pl|\.png|\.sh|\.txt)$$|checkdiag|COPYRIGHT|docs|example|make\.|Makefile|quark|README|Release|results|testing_|testing/lin|testing/matgen|tools' files.txt \
		| perl -pe 'chomp; $$_ = sprintf("\t../%-57s\\\n", $$_);' \
		> files-doxygen.txt

# files.txt per sub-directory
subdir_files = $(addsuffix /files.txt,$(subdirs) $(sparse_subdirs))

$(subdir_files): force
	svn st -N -vq $(dir $@) \
		| egrep -v '^D|> moved' \
		| perl -pi -e 's%^.{13} +\S+ +\S+ +\S+ +$(dir $@)%%' | sort \
		| egrep -v '^\.$$|obsolete|deprecated|contrib\b|^exp' \
		| egrep '\w\.\w|Makefile|docs|run' \
		> $@


# ----------------------------------------
echo:
	@echo "====="
	@echo "shdr   $(shdr)\n"
	@echo "dhdr   $(dhdr)\n"
	@echo "chdr   $(chdr)\n"
	@echo "zhdr   $(zhdr)\n"
	@echo "allhdr $(allhdr)\n"
	@echo "====="
	@echo "libmagma_ssrc   $(libmagma_ssrc)\n"
	@echo "libmagma_dsrc   $(libmagma_dsrc)\n"
	@echo "libmagma_csrc   $(libmagma_csrc)\n"
	@echo "libmagma_zsrc   $(libmagma_zsrc)\n"
	@echo "libmagma_src    $(libmagma_src)\n"
	@echo "libmagma_allsrc $(libmagma_allsrc)\n"
	@echo "libmagma_obj    $(libmagma_obj)\n"
	@echo "libmagma_a      $(libmagma_a)"
	@echo "libmagma_so     $(libmagma_so)"
	@echo "====="
	@echo "libsparse_ssrc   $(libsparse_ssrc)\n"
	@echo "libsparse_dsrc   $(libsparse_dsrc)\n"
	@echo "libsparse_csrc   $(libsparse_csrc)\n"
	@echo "libsparse_zsrc   $(libsparse_zsrc)\n"
	@echo "libsparse_src    $(libsparse_src)\n"
	@echo "libsparse_allsrc $(libsparse_allsrc)\n"
	@echo "libsparse_obj    $(libsparse_obj)\n"
	@echo "libsparse_a      $(libsparse_a)"
	@echo "libsparse_so     $(libsparse_so)"
	@echo "====="
	@echo "blas_fix        $(blas_fix)"
	@echo "libblas_fix_src $(libblas_fix_src)"
	@echo "libblas_fix_a   $(libblas_fix_a)"
	@echo "====="
	@echo "libtest_ssrc    $(libtest_ssrc)\n"
	@echo "libtest_dsrc    $(libtest_dsrc)\n"
	@echo "libtest_csrc    $(libtest_csrc)\n"
	@echo "libtest_zsrc    $(libtest_zsrc)\n"
	@echo "libtest_src     $(libtest_src)\n"
	@echo "libtest_allsrc  $(libtest_allsrc)\n"
	@echo "libtest_obj     $(libtest_obj)\n"
	@echo "libtest_a       $(libtest_a)\n"
	@echo "====="
	@echo "liblapacktest_ssrc    $(liblapacktest_ssrc)\n"
	@echo "liblapacktest_dsrc    $(liblapacktest_dsrc)\n"
	@echo "liblapacktest_csrc    $(liblapacktest_csrc)\n"
	@echo "liblapacktest_zsrc    $(liblapacktest_zsrc)\n"
	@echo "liblapacktest_src     $(liblapacktest_src)\n"
	@echo "liblapacktest_allsrc  $(liblapacktest_allsrc)\n"
	@echo "liblapacktest_obj     $(liblapacktest_obj)\n"
	@echo "liblapacktest_a       $(liblapacktest_a)\n"
	@echo "====="
	@echo "testing_ssrc    $(testing_ssrc)\n"
	@echo "testing_dsrc    $(testing_dsrc)\n"
	@echo "testing_csrc    $(testing_csrc)\n"
	@echo "testing_zsrc    $(testing_zsrc)\n"
	@echo "testing_src     $(testing_src)\n"
	@echo "testing_allsrc  $(testing_allsrc)\n"
	@echo "testing_obj     $(testing_obj)\n"
	@echo "testers         $(testers)\n"
	@echo "testers_f       $(testers_f)\n"
	@echo "====="
	@echo "sparse_testing_ssrc    $(sparse_testing_ssrc)\n"
	@echo "sparse_testing_dsrc    $(sparse_testing_dsrc)\n"
	@echo "sparse_testing_csrc    $(sparse_testing_csrc)\n"
	@echo "sparse_testing_zsrc    $(sparse_testing_zsrc)\n"
	@echo "sparse_testing_src     $(sparse_testing_src)\n"
	@echo "sparse_testing_allsrc  $(sparse_testing_allsrc)\n"
	@echo "sparse_testing_obj     $(sparse_testing_obj)\n"
	@echo "sparse_testers         $(sparse_testers)\n"
	@echo "====="
	@echo "dep     $(dep)"
	@echo "deps    $(deps)\n"
	@echo "====="
	@echo "libs    $(libs)"
	@echo "libs_a  $(libs_a)"
	@echo "libs_so $(libs_so)"
	@echo "====="
	@echo "LIBEXT  $(LIBEXT)"


# ----------------------------------------
cleandep:
	-rm -f $(deps)

ifeq ($(dep),1)
    -include $(deps)
endif
