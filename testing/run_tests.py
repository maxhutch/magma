#!/usr/bin/env python
#
# Script to run testers with various matrix sizes
# Small sizes are chosen around block sizes (e.g., 30...34 around 32) to
# detect bugs that occur at the block size, and the switch over from
# LAPACK to MAGMA code.
# Tall and wide sizes are chosen to exercise different aspect ratios,
# e.g., nearly square, 2:1, 10:1, 1:2, 1:10.
#
# By default, it runs all testers, pausing between each one.
# At the pause, typing "m" re-makes and re-runs that tester,
# while typing enter goes to the next tester.
# By default, only one combination of input parameters is tested, in most cases.
# The --first option runs only the first combination of input parameters (uplo, trans, etc.).
#
# The --start option skips all testers before the given one, then continues
# with testers from there.
#   ./run_tests.py --start testing_spotrf
#
# If specific testers are named on the command line, only those are run.
#   ./run_tests.py testing_spotrf testing_sgetrf
#
# The -p option controls what precisions are tested, the default being "sdcz"
# for all precisions.
#   ./run_tests.py -p sd
#
# The -s -m -l options control what sizes are tested, the default being
# all three sets.
# -s are small  tests, N < 300
# -m are medium tests, N < 1000
# -l are large  tests, N > 1000
#   ./run_tests.py -s -m
#
# @author Mark Gates

import os
import re
import sys
import time

import subprocess
from subprocess import PIPE, STDOUT

from optparse import OptionParser

# on a TTY screen, stop after each test for user input
# when redirected to file ("batch mode"), don't stop
batch = not sys.stdout.isatty()

parser = OptionParser()
parser.add_option('-p', '--precisions', action='store',      dest='precisions', help='run given precisions (initials, e.g., "sd" for single and double)', default='sdcz')
parser.add_option(      '--start',      action='store',      dest='start',      help='start with given routine; useful if run is interupted')
parser.add_option(      '--first',      action='store_true', dest='first',      help='run only the first combination of options for each tester')
parser.add_option(      '--memcheck',   action='store_true', dest='memcheck',   help='run with cuda-memcheck (slow)')
parser.add_option(      '--tol',        action='store',      dest='tol',        help='set tolerance')

parser.add_option('-s', '--small',      action='store_true', dest='small',      help='run small  tests, N < 300')
parser.add_option('-m', '--medium',     action='store_true', dest='med',        help='run medium tests, N < 1000')
parser.add_option('-l', '--large',      action='store_true', dest='large',      help='run large  tests, N > 1000')

parser.add_option(      '--blas',       action='store_true', dest='blas',       help='run BLAS tests')
parser.add_option(      '--aux',        action='store_true', dest='aux',        help='run auxiliary routine tests')
parser.add_option(      '--chol',       action='store_true', dest='chol',       help='run Cholesky factorization & solver tests')
parser.add_option(      '--lu',         action='store_true', dest='lu',         help='run LU factorization & solver tests')
parser.add_option(      '--qr',         action='store_true', dest='qr',         help='run QR factorization & solver (gels) tests')
parser.add_option(      '--syev',       action='store_true', dest='syev',       help='run symmetric eigenvalue tests')
parser.add_option(      '--geev',       action='store_true', dest='geev',       help='run non-symmetric eigenvalue tests')
parser.add_option(      '--svd',        action='store_true', dest='svd',        help='run SVD tests')

(opts, args) = parser.parse_args()

# default if no sizes given is all sizes
if ( not opts.small and not opts.med and not opts.large ):
	opts.small = True
	opts.med   = True
	opts.large = True
# end

# default if no groups given is all groups
if ( not opts.blas and not opts.aux  and
	 not opts.chol and not opts.lu   and not opts.qr   and
	 not opts.syev and not opts.geev and not opts.svd ):
	opts.blas = True
	opts.aux  = True
	opts.chol = True
	opts.lu   = True
	opts.qr   = True
	opts.syev = True
	opts.geev = True
	opts.svd  = True
# end

print 'opts', opts
print 'args', args


# ----------------------------------------------------------------------
# problem sizes
# n    is square
# tall is M > N
# wide is M < N
# mn   is all of above
# mnk  is all of above + combinations of M, N, K where K is unique

# ----------
n = ''
if opts.small:
	n += (' --range 1:20:1'
	  +   ' -N  30  -N  31  -N  32  -N  33  -N  34'
	  +   ' -N  62  -N  63  -N  64  -N  65  -N  66'
	  +   ' -N  94  -N  95  -N  96  -N  97  -N  98'
	  +   ' -N 126  -N 127  -N 128  -N 129  -N 130'
	  +   ' -N 254  -N 255  -N 256  -N 257  -N 258'
	)
if opts.med:
	n +=  ' -N 510  -N 511  -N 512  -N 513  -N 514 --range 100:900:100'
	n_sm = n
if opts.large:
	n +=  ' --range 1000:4000:1000'


# ----------
tall = ''
if opts.small:
	tall += (' -N 2,1        -N 3,1        -N 4,2'
	     +   ' -N 20,19      -N 20,10      -N 20,2      -N 20,1'
	     +   ' -N 200,199    -N 200,100    -N 200,20    -N 200,10    -N 200,1'
	)
if opts.med:
	tall +=  ' -N 600,599    -N 600,300    -N 600,60    -N 600,30    -N 600,10   -N 600,1'
if opts.large:
	tall +=  ' -N 2000,1999  -N 2000,1000  -N 2000,200  -N 2000,100  -N 2000,10  -N 2000,1'


# ----------
wide = ''
if opts.small:
	wide += (' -N 1,2        -N 1,3        -N 2,4'
	     +   ' -N 19,20      -N 10,20      -N 2,20      -N 1,20'
	     +   ' -N 199,200    -N 100,200    -N 20,200    -N 10,200    -N 1,200'
	)
if opts.med:
	wide +=  ' -N 599,600    -N 300,600    -N 60,600    -N 30,600    -N 10,600   -N 1,600'
if opts.large:
	wide +=  ' -N 1999,2000  -N 1000,2000  -N 200,2000  -N 100,2000  -N 10,2000  -N 1,2000'


# ----------
mnk = ''
if opts.small:
	mnk  += (' -N 1,2,3           -N 2,1,3           -N 1,3,2           -N 2,3,1           -N 3,1,2           -N 3,2,1'
	     +   ' -N 10,20,30        -N 20,10,30        -N 10,30,20        -N 20,30,10        -N 30,10,20        -N 30,20,10'
	     +   ' -N 100,200,300     -N 200,100,300     -N 100,300,200     -N 200,300,100     -N 300,100,200     -N 300,200,100'
	)
if opts.med:
	mnk  +=  ' -N 100,300,600     -N 300,100,600     -N 100,600,300     -N 300,600,100     -N 600,100,300     -N 600,300,100'
if opts.large:
	mnk  +=  ' -N 1000,2000,3000  -N 2000,1000,3000  -N 1000,3000,2000  -N 2000,3000,1000  -N 3000,1000,2000  -N 3000,2000,1000'


# ----------
mn     = n + tall + wide
mnk    = n + tall + wide + mnk


# ----------------------------------------------------------------------
# problems
# these match the order in the Makefile,
# except in some cases the "d" version isn't required here
#
# problems beginning with # are marked as disabled;
# the reason should be given in a comment afterward the item.

tests = []


# ----------
# BLAS
blas = (
	'./testing_z_cublas_v2           -c' + n,    # cublas only
	
	# no-trans/conj-trans; there are other combinations with trans
	'./testing_zgemm  -l -NN         -c' + mnk,
	'./testing_zgemm  -l -NC         -c' + mnk,
	'./testing_zgemm  -l -CN         -c' + mnk,
	'./testing_zgemm  -l -CC         -c' + mnk,
	
	'./testing_zgemv                 -c' + mn,
	'./testing_zgemv  -C             -c' + mn,
	
	# lower/upper
	'./testing_zhemv  -L             -c' + n,
	'./testing_zhemv  -U             -c' + n,
	
	# lower/upper, no-trans/conj-trans
	'./testing_zherk  -L             -c' + n,    # cublas only
	'./testing_zherk  -U -C          -c' + n,    # cublas only
	'./testing_zherk  -L             -c' + n,    # cublas only
	'./testing_zherk  -U -C          -c' + n,    # cublas only
	
	# lower/upper, no-trans/conj-trans
	'./testing_zher2k -L             -c' + n,    # cublas only
	'./testing_zher2k -L -C          -c' + n,    # cublas only
	'./testing_zher2k -U             -c' + n,    # cublas only
	'./testing_zher2k -U -C          -c' + n,    # cublas only
	
	# lower/upper
	'./testing_zsymv  -L             -c' + n,
	'#./testing_zsymv  -U             -c' + n,   # upper not implemented
	
	# left/right, lower/upper, no-trans/conj-trans, non-unit/unit diag
	'./testing_ztrmm  -SL -L    -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SL -L    -DU  -c' + mn,   # cublas only
	'./testing_ztrmm  -SL -L -C -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SL -L -C -DU  -c' + mn,   # cublas only
	
	'./testing_ztrmm  -SL -U    -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SL -U    -DU  -c' + mn,   # cublas only
	'./testing_ztrmm  -SL -U -C -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SL -U -C -DU  -c' + mn,   # cublas only
	
	'./testing_ztrmm  -SR -L    -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SR -L    -DU  -c' + mn,   # cublas only
	'./testing_ztrmm  -SR -L -C -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SR -L -C -DU  -c' + mn,   # cublas only
	
	'./testing_ztrmm  -SR -U    -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SR -U    -DU  -c' + mn,   # cublas only
	'./testing_ztrmm  -SR -U -C -DN  -c' + mn,   # cublas only
	'./testing_ztrmm  -SR -U -C -DU  -c' + mn,   # cublas only
	
	# lower/upper, no-trans/conj-trans, non-unit/unit diag
	'./testing_ztrmv      -L    -DN  -c' + n,    # cublas only
	'./testing_ztrmv      -L    -DU  -c' + n,    # cublas only
	'./testing_ztrmv      -L -C -DN  -c' + n,    # cublas only
	'./testing_ztrmv      -L -C -DU  -c' + n,    # cublas only
	
	'./testing_ztrmv      -U    -DN  -c' + n,    # cublas only
	'./testing_ztrmv      -U    -DU  -c' + n,    # cublas only
	'./testing_ztrmv      -U -C -DN  -c' + n,    # cublas only
	'./testing_ztrmv      -U -C -DU  -c' + n,    # cublas only
	
	# left/right, lower/upper, no-trans/conj-trans, non-unit/unit diag
	'./testing_ztrsm  -SL -L    -DN  -c' + mn,
	'./testing_ztrsm  -SL -L    -DU  -c' + mn,
	'./testing_ztrsm  -SL -L -C -DN  -c' + mn,
	'./testing_ztrsm  -SL -L -C -DU  -c' + mn,
	
	'./testing_ztrsm  -SL -U    -DN  -c' + mn,
	'./testing_ztrsm  -SL -U    -DU  -c' + mn,
	'./testing_ztrsm  -SL -U -C -DN  -c' + mn,
	'./testing_ztrsm  -SL -U -C -DU  -c' + mn,
	
	'./testing_ztrsm  -SR -L    -DN  -c' + mn,
	'./testing_ztrsm  -SR -L    -DU  -c' + mn,
	'./testing_ztrsm  -SR -L -C -DN  -c' + mn,
	'./testing_ztrsm  -SR -L -C -DU  -c' + mn,
	
	'./testing_ztrsm  -SR -U    -DN  -c' + mn,
	'./testing_ztrsm  -SR -U    -DU  -c' + mn,
	'./testing_ztrsm  -SR -U -C -DN  -c' + mn,
	'./testing_ztrsm  -SR -U -C -DU  -c' + mn,
	
	# lower/upper, no-trans/conj-trans, non-unit/unit diag
	'./testing_ztrsv      -L    -DN  -c' + n,    # cublas only
	'./testing_ztrsv      -L    -DU  -c' + n,    # cublas only
	'./testing_ztrsv      -L -C -DN  -c' + n,    # cublas only
	'./testing_ztrsv      -L -C -DU  -c' + n,    # cublas only
	
	'./testing_ztrsv      -U    -DN  -c' + n,    # cublas only
	'./testing_ztrsv      -U    -DU  -c' + n,    # cublas only
	'./testing_ztrsv      -U -C -DN  -c' + n,    # cublas only
	'./testing_ztrsv      -U -C -DU  -c' + n,    # cublas only
	
	# lower/upper
	'./testing_ztrtri_diag        -L -c' + n,
	'./testing_ztrtri_diag        -U -c' + n,
	
	'./testing_zblas                 -c' + mnk,  # cublas only
)
if ( opts.blas ):
	tests += blas

# ----------
# auxiliary
aux = (
	'./testing_auxiliary             -c',
	'./testing_constants             -c',
	'./testing_parse_opts            -c',
	'./testing_zgeadd                -c' + mn,
	'./testing_zgeadd_batched        -c' + mn,
	'./testing_zlacpy                -c' + mn,   # TODO implement uplo
	'./testing_zlacpy_batched        -c' + mn,   # TODO implement uplo
	'./testing_zlag2c                -c' + mn,
	'./testing_zlange                -c' + mn,
	
	# lower/upper
	'./testing_zlanhe -L             -c' + n,
	'./testing_zlanhe -U             -c' + n,
	
	'./testing_zlarfg                -c' + n,
	'./testing_zlaset                -c' + mn,
	'./testing_zlaset_band           -c' + mn,
	'./testing_znan_inf              -c' + mn,
	'./testing_zprint                -c -N 10 -N 5,100 -N 100,5',
	
	# lower/upper
	'./testing_zsymmetrize -L        -c' + n,
	'./testing_zsymmetrize -U        -c' + n,
	
	# lower/upper
	'./testing_zsymmetrize_tiles -L  -c' + n,
	'./testing_zsymmetrize_tiles -U  -c' + n,
	
	'./testing_zswap                 -c' + n,
	'./testing_ztranspose            -c' + mn,
)
if ( opts.aux ):
	tests += aux

# ----------
# Cholesky, GPU interface
chol = (
	'./testing_zcposv_gpu  -L    -c' + n,
	'./testing_zcposv_gpu  -U    -c' + n,
	
	'./testing_zposv_gpu   -L    -c' + n,
	'./testing_zposv_gpu   -U    -c' + n,
	
	'./testing_zpotrf_gpu  -L   -c2' + n,
	'./testing_zpotrf_gpu  -U   -c2' + n,
	
	'./testing_zpotf2_gpu  -L    -c' + n + tall,
	'./testing_zpotf2_gpu  -U    -c' + n + tall,
	
	'./testing_zpotri_gpu  -L    -c' + n,
	'./testing_zpotri_gpu  -U    -c' + n,
	
	'./testing_zpotrf_mgpu -L    -c' + n,
	'./testing_zpotrf_mgpu -U    -c' + n,
	
# ----------
# Cholesky, CPU interface
	'./testing_zposv       -L    -c' + n,
	'./testing_zposv       -U    -c' + n,
	
	'./testing_zpotrf      -L    -c' + n,
	'./testing_zpotrf      -U    -c' + n,
	
	'./testing_zpotri      -L    -c' + n,
	'./testing_zpotri      -U    -c' + n,
)
if ( opts.chol ):
	tests += chol

# ----------
# LU, GPU interface
lu = (
	'./testing_zcgesv_gpu        -c' + n,
	'./testing_zgesv_gpu         -c' + n,
	'./testing_zgetrf_gpu       -c2' + n,
	'./testing_zgetf2_gpu        -c' + n + tall,
	'./testing_zgetri_gpu        -c' + n,
	'./testing_zgetrf_mgpu      -c2' + n,
	
# ----------
# LU, CPU interface
	'./testing_zgesv             -c' + n,
	'./testing_zgetrf           -c2' + n,
)
if ( opts.lu ):
	tests += lu

# ----------
# QR and least squares, GPU interface
# TODO qrf uses  -c2 ?
qr = (
	'./testing_zcgeqrsv_gpu      -c' + mn,
	'./testing_zgelqf_gpu        -c' + mn,
	'./testing_zgels_gpu         -c' + mn,
	'./testing_zgels3_gpu        -c' + mn,
	'./testing_zgegqr_gpu --version 1 -c' + mn,
	'./testing_zgegqr_gpu --version 2 -c' + mn,
	'./testing_zgegqr_gpu --version 3 -c' + mn,
	'./testing_zgegqr_gpu --version 4 -c' + mn,
	'#./testing_zgeqp3_gpu        -c' + mn,  # fails badly
	'./testing_zgeqr2_gpu        -c' + mn,
	'./testing_zgeqr2x_gpu       -c' + mn,
	'./testing_zgeqrf_gpu        -c' + mn,
	'./testing_zlarfb_gpu        -c' + mnk,
	'./testing_zungqr_gpu        -c' + mnk,
	'./testing_zunmqr_gpu        -c' + mnk,
	'./testing_zgeqrf_mgpu       -c' + mn,
	
# ----------
# QR, CPU interface
	'./testing_zgelqf            -c' + mn,
	'./testing_zgeqlf            -c' + mn,
	'./testing_zgeqp3            -c' + mn,
	'./testing_zgeqrf            -c' + mn,
	'./testing_zungqr            -c' + mnk,
	'./testing_zunmlq            -c' + mnk,
	'./testing_zunmql            -c' + mnk,
	'./testing_zunmqr            -c' + mnk,
	'./testing_zungqr_m          -c' + mnk,
)
if ( opts.qr ):
	tests += qr

# ----------
# symmetric eigenvalues, GPU interface
syev = (
	# no-vectors/vectors, lower/upper
	'#./testing_zheevd_gpu  -L -JN -c' + n,  # does dsyevd_gpu  # -c implies -JV
	'#./testing_zheevd_gpu  -U -JN -c' + n,  # does dsyevd_gpu  # -c implies -JV
	'./testing_zheevd_gpu  -L -JV -c' + n,  # does dsyevd_gpu
	'./testing_zheevd_gpu  -U -JV -c' + n,  # does dsyevd_gpu
	
	'./testing_zhetrd_gpu  -L     -c' + n,
	'./testing_zhetrd_gpu  -U     -c' + n,
	
	'./testing_zhetrd_mgpu -L     -c' + n,
	'./testing_zhetrd_mgpu -U     -c' + n,
	
# ----------
# symmetric eigenvalues, CPU interface
	'#./testing_zheevd      -L -JN -c' + n,  # does dsyevd  # -c implies -JV
	'#./testing_zheevd      -U -JN -c' + n,  # does dsyevd  # -c implies -JV
	'./testing_zheevd      -L -JV -c' + n,  # does dsyevd
	'./testing_zheevd      -U -JV -c' + n,  # does dsyevd
	
	'./testing_zhetrd      -L     -c' + n,
	'./testing_zhetrd      -U     -c' + n,
	
# ----------
# symmetric eigenvalues, 2-stage
	#'./testing_zhetrd_he2hb   -L -c' + n,      # NOT hetrd_he2hb -- callsy heevdx_2stage
	#'./testing_zhetrd_he2hb   -U -c' + n,      # NOT hetrd_he2hb -- callsy heevdx_2stage, upper not implemented
	
	'#./testing_zheevdx_2stage -L -JN -c' + n,  # -c implies -JV
	'#./testing_zheevdx_2stage -U -JN -c' + n,  # -c implies -JV
	'./testing_zheevdx_2stage -L -JV -c' + n,
	'#./testing_zheevdx_2stage -U -JV -c' + n,  # upper not implemented
	
	'./testing_zhetrd_he2hb_mgpu -L -c' + n,
	'#./testing_zhetrd_he2hb_mgpu -U -c' + n,   # upper not implemented
	
# ----------
# generalized symmetric eigenvalues
	# no-vector/vector, lower/upper, itypes
	'#./testing_zhegvd   -L -JN --itype 1 -c' + n,  # does dsygvd  # -c implies -JV
	'#./testing_zhegvd   -L -JN --itype 2 -c' + n,  # does dsygvd  # -c implies -JV
	'#./testing_zhegvd   -L -JN --itype 3 -c' + n,  # does dsygvd  # -c implies -JV
	
	'#./testing_zhegvd   -U -JN --itype 1 -c' + n,  # does dsygvd  # -c implies -JV
	'#./testing_zhegvd   -U -JN --itype 2 -c' + n,  # does dsygvd  # -c implies -JV
	'#./testing_zhegvd   -U -JN --itype 3 -c' + n,  # does dsygvd  # -c implies -JV
	
	'./testing_zhegvd   -L -JV --itype 1 -c' + n,  # does dsygvd
	'./testing_zhegvd   -L -JV --itype 2 -c' + n,  # does dsygvd
	'./testing_zhegvd   -L -JV --itype 3 -c' + n,  # does dsygvd
	
	'./testing_zhegvd   -U -JV --itype 1 -c' + n,  # does dsygvd
	'./testing_zhegvd   -U -JV --itype 2 -c' + n,  # does dsygvd
	'./testing_zhegvd   -U -JV --itype 3 -c' + n,  # does dsygvd
	
	# lower/upper, no-vector/vector, itypes
	'#./testing_zhegvd_m -L -JN --itype 1 -c' + n,  # -c implies -JV
	'#./testing_zhegvd_m -L -JN --itype 2 -c' + n,  # -c implies -JV
	'#./testing_zhegvd_m -L -JN --itype 3 -c' + n,  # -c implies -JV
	
	'#./testing_zhegvd_m -U -JN --itype 1 -c' + n,  # -c implies -JV
	'#./testing_zhegvd_m -U -JN --itype 2 -c' + n,  # -c implies -JV
	'#./testing_zhegvd_m -U -JN --itype 3 -c' + n,  # -c implies -JV
	
	'./testing_zhegvd_m -L -JV --itype 1 -c' + n,
	'./testing_zhegvd_m -L -JV --itype 2 -c' + n,
	'./testing_zhegvd_m -L -JV --itype 3 -c' + n,
	
	'#./testing_zhegvd_m -U -JV --itype 1 -c' + n,  # upper not implemented
	'#./testing_zhegvd_m -U -JV --itype 2 -c' + n,  # upper not implemented
	'#./testing_zhegvd_m -U -JV --itype 3 -c' + n,  # upper not implemented
	
	# lower/upper, no-vector/vector, itypes
	'#./testing_zhegvdx  -L -JN --itype 1 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx  -L -JN --itype 2 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx  -L -JN --itype 3 -c' + n,  # -c implies -JV
	
	'#./testing_zhegvdx  -U -JN --itype 1 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx  -U -JN --itype 2 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx  -U -JN --itype 3 -c' + n,  # -c implies -JV
	
	'./testing_zhegvdx  -L -JV --itype 1 -c' + n,
	'./testing_zhegvdx  -L -JV --itype 2 -c' + n,
	'./testing_zhegvdx  -L -JV --itype 3 -c' + n,
	
	'./testing_zhegvdx  -U -JV --itype 1 -c' + n,
	'./testing_zhegvdx  -U -JV --itype 2 -c' + n,
	'./testing_zhegvdx  -U -JV --itype 3 -c' + n,
	
	# lower/upper, no-vector/vector, itypes
	'#./testing_zhegvdx_2stage -L -JN --itype 1 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx_2stage -L -JN --itype 2 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx_2stage -L -JN --itype 3 -c' + n,  # -c implies -JV
	
	'#./testing_zhegvdx_2stage -U -JN --itype 1 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx_2stage -U -JN --itype 2 -c' + n,  # -c implies -JV
	'#./testing_zhegvdx_2stage -U -JN --itype 3 -c' + n,  # -c implies -JV
	
	'./testing_zhegvdx_2stage -L -JV --itype 1 -c' + n,
	'./testing_zhegvdx_2stage -L -JV --itype 2 -c' + n,
	'./testing_zhegvdx_2stage -L -JV --itype 3 -c' + n,
	
	'#./testing_zhegvdx_2stage -U -JV --itype 1 -c' + n,  # upper not implemented
	'#./testing_zhegvdx_2stage -U -JV --itype 2 -c' + n,  # upper not implemented
	'#./testing_zhegvdx_2stage -U -JV --itype 3 -c' + n,  # upper not implemented
)
if ( opts.syev ):
	tests += syev

# ----------
# non-symmetric eigenvalues
geev = (
	# right & left no-vector/vector; not all combos are tested here
	'./testing_zgeev     -RN -LN -c' + n,  # does dgeev
	'./testing_zgeev     -RV -LV -c' + n,  # does dgeev
	
	'./testing_zgeev_m   -RN -LN -c' + n,  # does dgeev_m
	'./testing_zgeev_m   -RV -LV -c' + n,  # does dgeev_m
	
	'./testing_zgehrd            -c' + n,	
	'./testing_zgehrd_m          -c' + n,
)
if ( opts.geev ):
	tests += geev

# ----------
# SVD
svd = (
	# U & V none/some/overwrite/all
	# gesdd only has one jobz (taken from -U), while
	# gesvd can set U & V independently; not all combos are tested here
	'./testing_zgesdd    -UN     -c' + mn,
	'./testing_zgesdd    -US     -c' + mn,
	'./testing_zgesdd    -UO     -c' + mn,
	'./testing_zgesdd    -UA     -c' + mn,
	
	'./testing_zgesvd    -UN -VN -c' + mn,
	'./testing_zgesvd    -US -VS -c' + mn,
	'./testing_zgesvd    -UO -VS -c' + mn,
	'./testing_zgesvd    -UA -VA -c' + mn,
	
	'./testing_zgebrd            -c' + mn,
	'./testing_zunmbr            -c' + mnk,
)
if ( opts.svd ):
	tests += svd


# ----------------------------------------------------------------------
precisions = (
	's', 'd', 'c', 'z'
)

subs = (
	('',              'dlag2s',      '',              'zlag2c'    ),
	('ssy',           'dsy',         'che',           'zhe'       ),
	('sor',           'dor',         'cun',           'zun'       ),
	('sy2sb',         'sy2sb',       'he2hb',         'he2hb'     ),
	('',              'testing_ds',  '',              'testing_zc'),
	('testing_s',     'testing_d',   'testing_c',     'testing_z' ),
	('lansy',         'lansy',       'lanhe',         'lanhe'     ),
)

# ----------
def substitute( txt, pfrom, pto ):
	if ( pfrom != pto ):
		ifrom = precisions.index( pfrom )
		ito   = precisions.index( pto )
		for sub in subs:
			txt = re.sub( sub[ifrom], sub[ito], txt )
	# end
	return txt
# end


# ----------------------------------------------------------------------
def run( cmd ):
	words = re.split( ' +', cmd )
	# stdout & stderr are merged
	p = subprocess.Popen( words, bufsize=1, stdout=PIPE, stderr=STDOUT )
	
	failures = 0
	for line in p.stdout:
		print line.rstrip()
		if re.search( 'fail|exit|memory mapping error|CUDA runtime error|illegal value|ERROR SUMMARY: [1-9]', line ):
			failures += 1
	
	err = p.wait()
	#print 'err', err, 'failures', failures
	if err == 0:
		return failures
	else:
		return err
# end


# ----------------------------------------------------------------------
nfail = 0
failures = []

start = None
if ( opts.start ):
	start = re.compile( opts.start + r'\b' )

seen  = {}
seen2 = {}
pause = 0

for test in tests:
	
	make = False
	for p in opts.precisions:
		if ( opts.tol ):
			test += ' --tolerance ' + opts.tol
		
		test2 = substitute( test, 'z', p )
		test2 = re.sub( '  +', ' ', test2 )  # compress spaces
		
		# skip tests before start
		if ( start and not start.search( test  )
		           and not start.search( test2 )):
			continue
		start = None
		
		# extract command "./testing_foo" and flags
		# (arguments through -c, excluding -N, --range) from the test
		# also check for disabled tests (beginning with #)
		disabled = False
		m = re.search( '^(#?)\./(\S*)( .*-c)?', test2 )
		if ( m ):
			disabled = m.group(1)
			cmd      = m.group(2)
			flags    = m.group(3) or ''
		else:
			print "** Internal error; test didn't match pattern! should be: ./testing_foo ... -c ...\n", test2
			continue
		# end
		
		# skip tests not in args, or duplicates, or non-existing
		if (    (args and not cmd in args)
		     or (not os.path.exists( cmd ))
		     or (opts.first and seen.has_key( cmd ))
		     or (seen2.has_key( test2 )) ):
			continue
		# end
		if ( not disabled ):
			seen[cmd] = True
		seen2[test2] = True
		
		if ( opts.memcheck ):
			test2 = 'cuda-memcheck ' + test2
		
		go = True
		while( go ):
			if pause > 0:
				time.sleep( pause )
				pause = 0
			# end
			
			print
			print '*'*100
			print test2
			print '*'*100
			sys.stdout.flush()
			
			sys.stderr.write( '%-40s' % (cmd + flags) )
			sys.stderr.flush()
			
			if ( disabled ):
				sys.stderr.write( '  (disabled)\n' )
				go = False
				continue
			# end
			
			if ( make ):
				m = 'make lib ' + cmd
				print m
				run( m )
			# end
			
			t = time.time()
			err = run( test2 )
			t = time.time() - t
			
			if ( err < 0 ):
				sys.stderr.write( '  ** failed: signal %d\n' % (-err) )
				sys.stdout.write( '  ** failed: signal %d\n' % (-err) )
				nfail += 1
				failures.append( cmd )
			elif ( err > 0 ):
				sys.stderr.write( '  ** %d tests failed\n' % (err) )
				sys.stdout.write( '  ** %d tests failed\n' % (err) )
				nfail += err
				failures.append( cmd )
			else:
				sys.stderr.write( '  ok\n' )
			# end
			
			if ( batch ):
				# sleep couple seconds to allow it to cool off some between tests
				pause = min( t, 2. )
				go = False
			else:
				x = raw_input( '[enter to continue; M to make and re-run] ' )
				if ( x in ('m','M')):
					make = True
				else:
					go = False
		# end
	# end
# end

print
print '*'*100
print 'summary'
print '*'*100

if ( nfail > 0 ):
	print nfail, 'tests failed in', len(failures), 'routines:'
	print '\t' + '\n\t'.join( failures )
else:
	print 'all tests passed'
