Getting started with MAGMA.

This is a simple, standalone example to show how to use MAGMA, once it is
compiled. More involved examples for individual routines are in the testing
directory. The testing code include some extra utilities that we use for
testing, such as testings.h and libtest.a, which are not required to use MAGMA,
though you may use them if desired.

If you want CUBLAS functions, include that header first, before magma.h:

    #include "cublas_v2.h"

Include the MAGMA header:

    #include "magma.h"

You may also need BLAS and LAPACK functions, which you can get with:

    #include "magma_lapack.h"

You can also use headers that came with your BLAS and LAPACK library, such as
Intel MKL. However, their definitions, while compatible, may not exactly match
ours. Especially their definition of the COMPLEX type will be different. We use
magmaDoubleComplex, which is a typedef of cuDoubleComplex. You may need to cast
back-and-forth between definitions.

When MAGMA was compiled, one of ADD_, NOCHANGE, or UPCASE was defined in
make.inc for how Fortran functions are name-mangled on your system. The most
common is ADD_, where Fortran adds an underscore after the function. Usually,
you can tell the convention by using nm to examine your BLAS library:

    bunsen ~> nm /mnt/scratch/openblas/lib/libopenblas.a | grep -i dnrm2
    dnrm2.o:
    0000000000000000 T dnrm2_
                     U dnrm2_k

Since dnrm2_ has an underscore, use ADD_. Then add the include paths. For
example, to compile your .c file:

    gcc -DADD_ -DHAVE_CUBLAS \
        -I$(MAGMADIR)/include -I$(CUDADIR)/include \
        -c -o example.c

where $(MAGMADIR) and $(CUDADIR) are set to where MAGMA and CUDA are installed,
respectively.

To link, add the library paths and necessary libraries. For example with OpenBLAS:

    gcc -L$(MAGMADIR)/lib -L$(CUDADIR)/lib -L$(OPENBLASDIR) \
        -lmagma -lcublas -lcudart -lopenblas \
        -o example example.o

The Makefile provided in this directory is a starting point for compiling. You
will need to adjust the MAGMA_CFLAGS and MAGMA_LIBS to reflect your system. See
the MAGMA make.inc file for which libraries are required for BLAS & LAPACK.

Alternatively, you can use pkg-config to get MAGMA's CFLAGS and LIBS
automatically. To use pkg-config, install MAGMA with 'make install', add MAGMA
to $PKG_CONFIG_PATH, e.g. with csh,

    setenv PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:/usr/local/magma/lib/pkgconfig

then use 'pkg-config --cflags magma' and 'pkg-config --libs magma' as shown in
the Makefile.
