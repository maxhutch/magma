#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2015
#//////////////////////////////////////////////////////////////////////////////

# GPU_TARGET contains one or more of Tesla, Fermi, or Kepler,
# to specify for which GPUs you want to compile MAGMA:
#     Tesla  - NVIDIA compute capability 1.x cards (no longer supported in CUDA 6.5)
#     Fermi  - NVIDIA compute capability 2.x cards
#     Kepler - NVIDIA compute capability 3.x cards
# The default is "Fermi Kepler".
# See http://developer.nvidia.com/cuda-gpus
#
#GPU_TARGET ?= Fermi Kepler

CC        = gcc
CXX       = g++
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC      = -fPIC

CFLAGS    = -O3 $(FPIC) -DADD_ -Wall -fopenmp -DMAGMA_SETAFFINITY -DMAGMA_WITH_ACML
FFLAGS    = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS  = -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
NVCCFLAGS = -O3         -DADD_       -Xcompiler "-fno-strict-aliasing $(FPIC)"
LDFLAGS   =     $(FPIC)              -fopenmp

# for ACML 4
#LIB      = -lacml_mp -lacml_mv -lcblas -lcublas -lcudart -lstdc++ -lm

# for ACML >= 5
# requires gcc & gfortran >= 4.6
LIB       = -lacml_mp -lcblas -lcublas -lcudart -lstdc++ -lm -lgfortran

# define library directories preferably in your environment, or here.
#ACMLDIR  ?= /opt/acml
#CBLASDIR ?= /opt/CBLAS
#CUDADIR  ?= /usr/local/cuda
-include make.check-acml
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \
            -L$(ACMLDIR)/gfortran64_mp/lib \
            -L$(CBLASDIR)/lib

INC       = -I$(CUDADIR)/include \
            -I$(ACMLDIR)/gfortran64_mp/include
