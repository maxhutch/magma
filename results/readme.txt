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

symv                 -- lower

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

Output is stored in the hierarchy below. To document results, please store the
setup.txt and make.inc files, in addition to data files.

magma version
	cuda version-gpu model
		setup.txt (host, GPU, CPU, gcc version, etc.)
		make.inc
		[sdcz]{getrf,potrf,geqrf,geev,syev,heev,gesvd,symv,...}.txt

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
