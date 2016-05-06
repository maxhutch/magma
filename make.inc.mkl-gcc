#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.0.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date May 2016
#//////////////////////////////////////////////////////////////////////////////

# GPU_TARGET contains one or more of Fermi, Kepler, or Maxwell,
# to specify for which GPUs you want to compile MAGMA:
#     Fermi   - NVIDIA compute capability 2.x cards
#     Kepler  - NVIDIA compute capability 3.x cards
#     Maxwell - NVIDIA compute capability 5.x cards
# The default is "Fermi Kepler".
# Note that NVIDIA no longer supports 1.x cards, as of CUDA 6.5.
# See http://developer.nvidia.com/cuda-gpus
#
#GPU_TARGET ?= Fermi Kepler

# --------------------
# programs

CC        = gcc
CXX       = g++
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib


# --------------------
# flags

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC      = -fPIC

CFLAGS    = -O3 $(FPIC) -DADD_ -Wall -Wshadow -fopenmp -DMAGMA_WITH_MKL 
FFLAGS    = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS  = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
NVCCFLAGS = -O3         -DADD_       -Xcompiler "$(FPIC) -Wall -Wno-unused-function"
LDFLAGS   =     $(FPIC)              -fopenmp

# C++11 (gcc >= 4.7) is not required, but has benefits like atomic operations
CXXFLAGS := $(CFLAGS) -std=c++11
CFLAGS   += -std=c99


# --------------------
# libraries

# see MKL Link Advisor at http://software.intel.com/sites/products/mkl/
# gcc with MKL 10.3, sequential version
#LIB       = -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lstdc++ -lm -lgfortran

# gcc with MKL 10.3, GNU threads
#LIB       = -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lstdc++ -lm -lgfortran

# gcc with MKL 10.3, Intel threads
LIB       = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lm -liomp5 -lgfortran

LIB      += -lcublas -lcusparse -lcudart


# --------------------
# directories

# define library directories preferably in your environment, or here.
# for MKL run, e.g.: source /opt/intel/composerxe/mkl/bin/mklvars.sh intel64
#MKLROOT ?= /opt/intel/composerxe/mkl
#CUDADIR ?= /usr/local/cuda
-include make.check-mkl
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \
            -L$(MKLROOT)/lib/intel64

INC       = -I$(CUDADIR)/include \
            -I$(MKLROOT)/include
