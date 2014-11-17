# This is a simple standalone example. See README.txt
# Initially it is setup to use OpenBLAS.
# See magma/make.inc for alternate BLAS and LAPACK libraries,
# or use pkg-config as described below.

# Paths where MAGMA, CUDA, and OpenBLAS are installed
MAGMADIR     ?= ..
CUDADIR      ?= /usr/local/cuda
OPENBLASDIR  ?= /usr/local/openblas

CC            = gcc
LD            = gcc
CFLAGS        = -Wall
LDFLAGS       = -Wall

MAGMA_CFLAGS := -DADD_ -I$(MAGMADIR)/include -I$(CUDADIR)/include

MAGMA_LIBS   := -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib \
                -lmagma -lcublas -lcudart -lopenblas

# Alternatively, using pkg-config (see README.txt):
# MAGMA_CFLAGS := $(shell pkg-config --cflags magma)
# MAGMA_LIBS   := $(shell pkg-config --libs   magma)

all: example

%.o: %.c
	$(CC) $(CFLAGS) $(MAGMA_CFLAGS) -c -o $@ $<

example: example.o zfill.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)

clean:
	-rm -f example *.o
