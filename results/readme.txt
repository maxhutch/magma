This contains results for various versions of MAGMA, CUDA, and GPU cards.

Routines currently tested are:
getrf, getrf_gpu
potrf, potrf_gpu     -- lower
geqrf, geqrf_gpu     -- square

geev                 -- no vectors, vectors
sy/heev, sy/heev_gpu -- no vectors, vectors
sy/heev_2stage       -- no vectors, vectors
gesvd                -- no vectors, some vectors; square, 3:1, 1:3, 100:1, 1:100
gesdd                -- no vectors, some vectors; square, 3:1, 1:3, 100:1, 1:100

symv                 -- lower, upper

TODO
extend potrf to upper
extend symv  to upper
extend geqrf to tall, wide

For most routines, this standard set of sizes is used:
small:  --range 10:90:10
medium: --range 100:900:100
large:  --range 1000:9000:1000
xlarge: --range 10000:20000:2000

For SVD, also tests 3:1, 1:3, 100:1, 1:100 ratios.
For symv, the small, medium, large ranges use steps of 1, 10, 100, respectively.



Running tests
=============
To compile MAGMA for running tests, use one of these make.inc files:
make.inc.mkl-icc-ilp64 (preferred) or
make.inc.mkl-ilp64     (older versions, with gcc).

To run tests, use include run*.csh scripts.
Each should generate [sdcz]xxxx.txt output files.

    ./run_amigos.csh
    ls [sdcz]*.txt
    cgeqrf.txt  dgeqrf.txt  sgeqrf.txt  zgeqrf.txt
    cgetrf.txt  dgetrf.txt  sgetrf.txt  zgetrf.txt
    cpotrf.txt  dpotrf.txt  spotrf.txt  zpotrf.txt

For local tests, simply parse data into local.py, which plots.py automatically
loads.

    ./parse.py [sdcz]*.txt > local.py

You may want to edit the version, device, and cpu meta-data in local.py.



Archiving results
=================
For archive purposes, output is stored in the hierarchy below. To document
results, please store the setup.txt and make.inc files, in addition to data
files.

magma version
    cuda version-gpu model
        setup.txt (host, GPU, CPU, gcc version, etc.)
        make.inc
        [sdcz]{getrf,potrf,...}.txt

Example:
v1.6.0
    cuda7.0-k40c
        setup.txt
        make.inc
        sgetrf.txt
        dgetrf.txt
        cgetrf.txt
        zgetrf.txt
        etc.

Parse the data into a Python data file, named for the version information.

    ./parse.py v1.6.0/cuda7.0-k40c/*.txt > v160_cuda70_k40c.py

In that Python data file, you need to edit the GPU and CPU information. These
should be set to match existing data files using the same hardware:

    version = '1.6.0'
    cuda    = '7.0'
    device  = 'Kepler K40c'                     # <== change
    cpu     = '2x8 core Sandy Bridge E5-2670'   # <== change

Add the data file to parse.py:

    import v150_cuda70_k40c
    import v160_cuda70_k40c  # <== add
    
    versions = [
        v150_cuda70_k40c,
        v160_cuda70_k40c,    # <== add
    ]

Add both the Python data file and the raw data files to SVN.

    svn add v160_cuda70_k40c.py
    svn add v1.6.0/



Plotting results
================
The plots.py Python script plots the performance of various routines from
different versions of MAGMA. It requires matplotlib (pyplot).

plots.versions is an array of available MAGMA versions; use array slices, as
shown below, to select which versions to plot.

plots.help() prints the global settings and available plots.

It is easiest to use plots.py from an interactive Python shell. IPython is
recommended.

Example using ipython:

    mint magma-trunk/results> ipython
    >>> run -i plots  # this prints help; -i is necessary so g_save, etc. are in the right name space to be modified
    
    Available versions:
    versions[0] = 1.5.0
    versions[1] = 1.6.0
    versions[2] = 1.6.1
    versions[3] = 1.6.2
    versions[4] = 1.7.0
    
    >>> plot_potrf( versions[ 4 ] )        # index 4 (1.7.0) only
    >>> plot_potrf( versions[ 2: ] )       # index 2 (1.6.1) and later
    >>> g_save = True                      # save PDF files
    >>> g_subplots = False                 # plot precisions as 4 figures instead of 4 subplots
    >>> plot_potrf( versions[ [0,2,4] ] )  # index 0 (1.5.0), 2 (1.6.1), and 4 (1.7.0)
    saving spotrf.pdf
    saving dpotrf.pdf
    saving cpotrf.pdf
    saving zpotrf.pdf

Using regular python:

    magma-trunk/results> python
    >>> import plots  # this prints help
    >>> plots.g_save = True
    >>> plots.plot_getrf( plots.versions[ [0,2,4] ] )
    saving getrf.pdf
    
    >>> plots.g_subplots = False
    >>> plots.plot_getrf( plots.versions[ [0,2,4] ] )
    saving sgetrf.pdf
    saving dgetrf.pdf
    saving cgetrf.pdf
    saving zgetrf.pdf
    
    >>> plots.help()
    Global settings:
    g_save      # True to save plots as PDF files
    g_subplots  # True for all 4 precisions as subplots in one figure, False for 4 separate figures
    g_log       # True for semilogx, False for linear plot
    g_figsize   # size of figure with 4-up subplots, default (9,7)
    g_figsize2  # size of individual figures, default (6,4)
    
    Available plots:
    plot_getrf( versions, lapack=True, cpu=True, gpu=True )
    plot_potrf( versions, lapack=True, cpu=True, gpu=True )
    plot_geqrf( versions, lapack=True, cpu=True, gpu=True )
    plot_geev(  versions, lapack=True )
    plot_syev(  versions, lapack=True, cpu=True, gpu=True, bulge=True )
    plot_gesvd( versions, lapack=True, svd=True, sdd=True, ratio=1 )
            where ratio m:n in { 1, 3, 100, 1/3., 1/100. }
    plot_symv(  versions, lapack=True )
    
    plot_all(   versions, lapack=True, cpu=True, gpu=True, bulge=True, sdd=True, svd=True, ratio=1 )
    
    Available versions:
    versions[0] = 1.5.0
    versions[1] = 1.6.0
    versions[2] = 1.6.1
    versions[3] = 1.6.2
    versions[4] = 1.7.0
