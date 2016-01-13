#!/usr/bin/env python
#
# MAGMA (version 2.0.0-beta2) --
# Univ. of Tennessee, Knoxville
# Univ. of California, Berkeley
# Univ. of Colorado, Denver
# @date January 2016

## @file run_tests.py
#  @author Mark Gates
#
# Script to run testers with various matrix sizes.
# Small sizes are chosen around block sizes (e.g., 30...34 around 32) to
# detect bugs that occur at the block size, and the switch over from
# LAPACK to MAGMA code.
# Tall and wide sizes are chosen to exercise different aspect ratios,
# e.g., nearly square, 2:1, 10:1, 1:2, 1:10.
# The -h or --help option provides a summary of the options.
#
# Non-interactive vs. interactive mode
# --------------------------
# When output is redirected to a file, it runs in non-interactive mode, printing a
# short summary to stderr on the console and all other output to the file.
# For example:
#
#       ./run_tests.py --lu --precision s --small > lu.txt
#       testing_sgesv_gpu -c                      ok
#       testing_sgetrf_gpu -c2                    ok
#       testing_sgetf2_gpu -c                     ok
#       testing_sgetri_gpu -c                     ** 45 tests failed
#       testing_sgetrf_mgpu -c2                   ok
#       testing_sgesv -c                          ok
#       testing_sgetrf -c2                        ok
#
#       ****************************************************************************************************
#       summary
#       ****************************************************************************************************
#         282 tests in 7 commands passed
#          45 tests failed accuracy test
#           0 errors detected (crashes, CUDA errors, etc.)
#       routines with failures:
#           testing_sgetri_gpu -c
#
# When output is to console (tty), it runs in interactive mode, pausing after
# each test. At the pause, typing "M" re-makes and re-runs that tester,
# while typing enter goes to the next tester.
# For example (some output suppressed with ... for brevity):
#
#       ./run_tests.py --lu --precision s --small
#       ****************************************************************************************************
#       ./testing_sgesv_gpu -c --range 1:20:1 ...
#       ****************************************************************************************************
#           N  NRHS   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||B - AX|| / N*||A||*||X||
#       ================================================================================
#           1     1     ---   (  ---  )      0.00 (   0.00)   9.26e-08   ok
#           2     1     ---   (  ---  )      0.00 (   0.00)   1.32e-08   ok
#           3     1     ---   (  ---  )      0.00 (   0.00)   8.99e-09   ok
#       ...
#         ok
#       [enter to continue; M to make and re-run]
#
#       ****************************************************************************************************
#       ./testing_sgetri_gpu -c --range 1:20:1 ...
#       ****************************************************************************************************
#       % MAGMA 1.4.0 svn compiled for CUDA capability >= 3.0
#       % CUDA runtime 6000, driver 6000. MAGMA not compiled with OpenMP.
#       % device 0: GeForce GT 750M, 925.5 MHz clock, 2047.6 MB memory, capability 3.0
#       Usage: ./testing_sgetri_gpu [options] [-h|--help]
#
#           N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / (N*||A||_F)
#       =================================================================
#           1      0.00 (   0.00)      0.00 (   0.00)   6.87e+01   failed
#           2      0.00 (   0.00)      0.00 (   0.00)   2.41e+00   failed
#           3      0.01 (   0.00)      0.00 (   0.00)   1.12e+00   failed
#       ...
#         ** 45 tests failed
#       [enter to continue; M to make and re-run]
#
#       ...
#
#       ****************************************************************************************************
#       summary
#       ****************************************************************************************************
#         282 tests in 7 commands passed
#          45 tests failed accuracy test
#           0 errors detected (crashes, CUDA errors, etc.)
#       routines with failures:
#           testing_sgetri_gpu -c
#
#
# What tests are run
# ------------------
# The --blas, --aux, --chol, --hesv, --lu, --qr, --syev, --sygv, --geev, --svd,
# --batched options run particular sets of tests. By default, all tests are run,
# except batched because we don't want to run batched with, say, N=1000.
# --mgpu runs only multi-GPU tests from the above sets.
#
# The --start option skips all testers before the given one, then continues
# with testers from there. This is helpful to restart a non-interactive set
# of tests. For example:
#
#       ./run_tests.py --start testing_spotrf > output.log
#
# If specific testers are named on the command line, only those are run.
# For example:
#
#       ./run_tests.py testing_spotrf testing_sgetrf
#
# The -p/--precision option controls what precisions are tested, the default
# being "sdcz" for all four precisions. For example, to run single and double:
#
#       ./run_tests.py -p sd
#
# The -s/--small, -m/--medium, -l/--large options control what sizes are tested,
# the default being all three sets.
#       -s/--small  does small  tests, N < 300.
#       -m/--medium does medium tests, N < 1000.
#       -l/--large  does large  tests, N > 1000.
# For example, running small and medium tests:
#
#       ./run_tests.py -s -m
#
#
# What is checked
# ------------------
# The --memcheck option runs cuda-memcheck. This is very helpful for finding
# memory bugs (reading & writing outside allocated memory). It is, however, slow.
#
# The --tol option sets the tolerance to verify accuracy. This is 30 by default,
# which may be too tight for some testers. Setting it somewhat higher
# (e.g., 50 or 100) filters out spurious accuracy failures.
#
# The --dev option sets which GPU device to use.

import os
import re
import sys
import time

import subprocess
from subprocess import PIPE, STDOUT

from optparse import OptionParser

# on a TTY screen, stop after each test for user input
# when redirected to file ("non-interactive mode"), don't stop
non_interactive = not sys.stdout.isatty()

parser = OptionParser()
parser.add_option('-p', '--precisions', action='store',      dest='precisions', help='run given precisions (initials, e.g., "sd" for single and double)', default='sdcz')
parser.add_option(      '--start',      action='store',      dest='start',      help='start with given routine; useful to restart an interupted run')
parser.add_option(      '--memcheck',   action='store_true', dest='memcheck',   help='run with cuda-memcheck (slow)')
parser.add_option(      '--tol',        action='store',      dest='tol',        help='set tolerance')
parser.add_option(      '--dev',        action='store',      dest='dev',        help='set GPU device to use')
parser.add_option(      '--batch',      action='store',      dest='batch',      help='batch count for batched tests', default='100')
parser.add_option(      '--niter',      action='store',      dest='niter',      help='number of iterations to repeat', default='1')
parser.add_option(      '--ngpu',       action='store',      dest='ngpu',       help='number of GPUs for multi-GPU tests; add --mgpu to run only multi-GPU tests', default='2')
parser.add_option(      '--null-stream',action='store_true', dest='null_stream',help='use null stream (for verifying old behavior)')

# options to specify sizes
parser.add_option(      '--xsmall',     action='store_true', dest='xsmall',     help='run very few, extra small tests, N=25:100:25, 32:128:32')
parser.add_option('-s', '--small',      action='store_true', dest='small',      help='run small  tests, N < 300')
parser.add_option('-m', '--medium',     action='store_true', dest='med',        help='run medium tests, N < 1000')
parser.add_option('-l', '--large',      action='store_true', dest='large',      help='run large  tests, N > 1000')
parser.add_option('-N',                 action='append',     dest='N',          help='run specific sizes; repeatable', default=[])
parser.add_option(      '--range',      action='append',     dest='range',      help='run specific sizes; repeatable', default=[])

# options to select classes of routines
parser.add_option(      '--blas',       action='store_true', dest='blas',       help='run BLAS tests')
parser.add_option(      '--aux',        action='store_true', dest='aux',        help='run auxiliary routine tests')
parser.add_option(      '--chol',       action='store_true', dest='chol',       help='run Cholesky factorization & solver tests')
parser.add_option(      '--hesv',       action='store_true', dest='hesv',       help='run Cholesky factorization & solver tests')
parser.add_option(      '--lu',         action='store_true', dest='lu',         help='run LU factorization & solver tests')
parser.add_option(      '--qr',         action='store_true', dest='qr',         help='run QR factorization & solver (gels) tests')
parser.add_option(      '--syev',       action='store_true', dest='syev',       help='run symmetric eigenvalue tests')
parser.add_option(      '--sygv',       action='store_true', dest='sygv',       help='run generalized symmetric eigenvalue tests')
parser.add_option(      '--geev',       action='store_true', dest='geev',       help='run non-symmetric eigenvalue tests')
parser.add_option(      '--svd',        action='store_true', dest='svd',        help='run SVD tests')
parser.add_option(      '--batched',    action='store_true', dest='batched',    help='run batched (BLAS, LU, etc.) tests')
parser.add_option(      '--mgpu',       action='store_true', dest='mgpu',       help='run multi-GPU (BLAS, LU, etc.) tests; add --ngpu to specify number of GPUs')
parser.add_option(      '--no-mgpu',    action='store_true', dest='no_mgpu',    help='do not run multi-GPU (BLAS, LU, etc.) tests')

# options to select subset of commands
parser.add_option(      '--itype',      action='store',      dest='itype',      help='select runs matching itype',   default=0 )
parser.add_option(      '--version',    action='store',      dest='version',    help='select runs matching version', default=0 )
parser.add_option('-U', '--upper',      action='store_true', dest='upper',      help='select runs matching upper',   default=None )
parser.add_option('-L', '--lower',      action='store_true', dest='lower',      help='select runs matching lower',   default=None )
parser.add_option('-J', '--jobz',       action='store',      dest='jobz',       help='select runs matching jobz (-JV, -JN)', default=None )
parser.add_option('-D', '--diag',       action='store',      dest='diag',       help='select runs matching diag (-DU, -DN)', default=None )
parser.add_option(      '--fraction',   action='store',      dest='fraction',   help='select runs matching diag (-DU, -DN)', default=None )

(opts, args) = parser.parse_args()

# default if no sizes given is all sizes (small, medium, large)
if ( not opts.xsmall and not opts.small and not opts.med and not opts.large ):
	opts.small = True
	opts.med   = True
	opts.large = True
# end

# default if no groups given is all groups
if ( not opts.blas and not opts.aux  and
	 not opts.chol and not opts.hesv and not opts.lu   and not opts.qr   and
	 not opts.syev and not opts.sygv and not opts.geev and
	 not opts.svd  and not opts.batched and
	 not opts.mgpu ):
	opts.blas = True
	opts.aux  = True
	opts.chol = True
	opts.hesv = True
	opts.lu   = True
	opts.qr   = True
	opts.syev = True
	opts.sygv = True
	opts.geev = True
	opts.svd  = True
	opts.batched = False   # batched routines must be explicitly requested, as the typical size range is different
	opts.mgpu    = False   # multi-GPU routines are part of above groups
# end

print 'opts', opts
print 'args', args

ngpu  = '--ngpu '  + opts.ngpu  + ' '
batch = '--batch ' + opts.batch + ' '

# ----------------------------------------------------------------------
# problem sizes
# n    is square
# tall is M > N
# wide is M < N
# mn   is all of above
# mnk  is all of above + combinations of M, N, K where K is unique

# ----------
n = ''
if opts.xsmall:
	n +=  ' --range 32:128:32 --range 25:100:25'
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
if opts.large:
	n +=  ' --range 1000:4000:1000'


# ----------
# to avoid excessive runtime with large m or n in zunmql, etc., k is set to min(m,n)
tall = ''
if opts.small:
	tall += (' -N 2,1        -N 3,1        -N 4,2'
	     +   ' -N 20,19      -N 20,10      -N 20,2      -N 20,1'
	     +   ' -N 200,199    -N 200,100    -N 200,20    -N 200,10    -N 200,1'
	)
if opts.med:
	tall += (' -N 600,599       -N 600,300       -N 10000,63,63   -N 10000,64,64   -N 10000,65,65'
         +   ' -N 10000,31,31   -N 10000,32,32   -N 10000,33,33   -N 10000,10,10   -N 10000,1,1')
if opts.large:
	tall +=  ' -N 2000,1999  -N 2000,1000  -N 20000,200,200  -N 20000,100,100  -N 200000,10,10  -N 200000,1,1  -N 2000000,10,10  -N 2000000,1,1'


# ----------
# to avoid excessive runtime with large m or n in zunmql, etc., k is set to min(m,n)
wide = ''
if opts.small:
	wide += (' -N 1,2        -N 1,3        -N 2,4'
	     +   ' -N 19,20      -N 10,20      -N 2,20      -N 1,20'
	     +   ' -N 199,200    -N 100,200    -N 20,200    -N 10,200    -N 1,200'
	)
if opts.med:
	wide += (' -N 599,600       -N 300,600       -N 63,10000,63   -N 64,10000,64   -N 65,10000,65'
         +   ' -N 31,10000,31   -N 32,10000,32   -N 33,10000,33   -N 10,10000,10   -N 1,10000,1')
if opts.large:
	wide +=  ' -N 1999,2000  -N 1000,2000  -N 200,20000,200  -N 100,20000,100  -N 10,200000,10  -N 1,200000,1  -N 10,2000000,10  -N 1,2000000,1'


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

# ----------
# specific sizes override everything else
print 'opts.N', opts.N
print 'opts.range', opts.range
if ( opts.N or opts.range ):
	n    = ' '.join( map( lambda x: '-N '+x, opts.N ) + map( lambda x: '--range '+x, opts.range ))
	mn   = n
	mnk  = n
	tall = ''
	wide = ''
# endif


# ----------------------------------------------------------------------
# test problems
#
# Each test has 4 fields:  (program, options, sizes, comments).
# If program begins with #, that test is disabled and will be printed as such.
# In some cases, the line is commented out, so nothing is printed.
#
# These match the order in the Makefile,
# except in some cases the "d" version from the Makefile isn't required here
# To compare with Makefile:
# grep '^\s+testing_\w+' Makefile >! a
# grep "\('#?testing_\w\w+" run_tests.py | perl -pe "s/#?\('#?//; s/',.*/.cpp \\/;" | uniq > ! b
# diff -w a b

tests = []


blas = (
	# ----------
	# BLAS
	# no-trans/conj-trans; there are other combinations with trans
	('testing_zgemm',   '-l -NN         -c',  mnk,  ''),
	('testing_zgemm',   '-l -NC         -c',  mnk,  ''),
	('testing_zgemm',   '-l -CN         -c',  mnk,  ''),
	('testing_zgemm',   '-l -CC         -c',  mnk,  ''),
	
	# no-trans/trans/conj-trans
	('testing_zgemv',                  '-c',  mn,   ''),
	('testing_zgemv',   '-T             -c',  mn,   ''),
	('testing_zgemv',   '-C             -c',  mn,   ''),
	
	# lower/upper
	('testing_zhemv',   '-L             -c',  n,    ''),
	('testing_zhemv',   '-U             -c',  n,    ''),
	
	# lower/upper, no-trans/conj-trans
	('testing_zherk',   '-L             -c',  n,    'cublas only'),
	('testing_zherk',   '-L -C          -c',  n,    'cublas only'),
	('testing_zherk',   '-U             -c',  n,    'cublas only'),
	('testing_zherk',   '-U -C          -c',  n,    'cublas only'),
	
	# lower/upper, no-trans/conj-trans
	('testing_zher2k',  '-L             -c',  n,    'cublas only'),
	('testing_zher2k',  '-L -C          -c',  n,    'cublas only'),
	('testing_zher2k',  '-U             -c',  n,    'cublas only'),
	('testing_zher2k',  '-U -C          -c',  n,    'cublas only'),
	
	# lower/upper
	('testing_zsymv',   '-L             -c',  n,    ''),
	('testing_zsymv',   '-U             -c',  n,    ''),
	
	# left/right, lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrmm',   '-SL -L    -DN  -c',  n + wide,   'cublas only'),
	('testing_ztrmm',   '-SL -L    -DU  -c',  n + wide,   'cublas only'),
	('testing_ztrmm',   '-SL -L -C -DN  -c',  n + wide,   'cublas only'),
	('testing_ztrmm',   '-SL -L -C -DU  -c',  n + wide,   'cublas only'),
	
	('testing_ztrmm',   '-SL -U    -DN  -c',  n + wide,   'cublas only'),
	('testing_ztrmm',   '-SL -U    -DU  -c',  n + wide,   'cublas only'),
	('testing_ztrmm',   '-SL -U -C -DN  -c',  n + wide,   'cublas only'),
	('testing_ztrmm',   '-SL -U -C -DU  -c',  n + wide,   'cublas only'),
	
	('testing_ztrmm',   '-SR -L    -DN  -c',  n + tall,   'cublas only'),
	('testing_ztrmm',   '-SR -L    -DU  -c',  n + tall,   'cublas only'),
	('testing_ztrmm',   '-SR -L -C -DN  -c',  n + tall,   'cublas only'),
	('testing_ztrmm',   '-SR -L -C -DU  -c',  n + tall,   'cublas only'),
	
	('testing_ztrmm',   '-SR -U    -DN  -c',  n + tall,   'cublas only'),
	('testing_ztrmm',   '-SR -U    -DU  -c',  n + tall,   'cublas only'),
	('testing_ztrmm',   '-SR -U -C -DN  -c',  n + tall,   'cublas only'),
	('testing_ztrmm',   '-SR -U -C -DU  -c',  n + tall,   'cublas only'),
	
	# lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrmv',       '-L    -DN  -c',  n,    'cublas only'),
	('testing_ztrmv',       '-L    -DU  -c',  n,    'cublas only'),
	('testing_ztrmv',       '-L -C -DN  -c',  n,    'cublas only'),
	('testing_ztrmv',       '-L -C -DU  -c',  n,    'cublas only'),
	
	('testing_ztrmv',       '-U    -DN  -c',  n,    'cublas only'),
	('testing_ztrmv',       '-U    -DU  -c',  n,    'cublas only'),
	('testing_ztrmv',       '-U -C -DN  -c',  n,    'cublas only'),
	('testing_ztrmv',       '-U -C -DU  -c',  n,    'cublas only'),
	
	# left/right, lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrsm',   '-SL -L    -DN  -c',  n + wide,   ''),
	('testing_ztrsm',   '-SL -L    -DU  -c',  n + wide,   ''),
	('testing_ztrsm',   '-SL -L -C -DN  -c',  n + wide,   ''),
	('testing_ztrsm',   '-SL -L -C -DU  -c',  n + wide,   ''),
	
	('testing_ztrsm',   '-SL -U    -DN  -c',  n + wide,   ''),
	('testing_ztrsm',   '-SL -U    -DU  -c',  n + wide,   ''),
	('testing_ztrsm',   '-SL -U -C -DN  -c',  n + wide,   ''),
	('testing_ztrsm',   '-SL -U -C -DU  -c',  n + wide,   ''),
	
	('testing_ztrsm',   '-SR -L    -DN  -c',  n + tall,   ''),
	('testing_ztrsm',   '-SR -L    -DU  -c',  n + tall,   ''),
	('testing_ztrsm',   '-SR -L -C -DN  -c',  n + tall,   ''),
	('testing_ztrsm',   '-SR -L -C -DU  -c',  n + tall,   ''),
	
	('testing_ztrsm',   '-SR -U    -DN  -c',  n + tall,   ''),
	('testing_ztrsm',   '-SR -U    -DU  -c',  n + tall,   ''),
	('testing_ztrsm',   '-SR -U -C -DN  -c',  n + tall,   ''),
	('testing_ztrsm',   '-SR -U -C -DU  -c',  n + tall,   ''),
	
	# left/right, lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrsm', ngpu + '-SL -L    -DN  -c',  n + wide,  ''),
	('testing_ztrsm', ngpu + '-SL -L    -DU  -c',  n + wide,  ''),
	('testing_ztrsm', ngpu + '-SL -L -C -DN  -c',  n + wide,  ''),
	('testing_ztrsm', ngpu + '-SL -L -C -DU  -c',  n + wide,  ''),
	
	('testing_ztrsm', ngpu + '-SL -U    -DN  -c',  n + wide,  ''),
	('testing_ztrsm', ngpu + '-SL -U    -DU  -c',  n + wide,  ''),
	('testing_ztrsm', ngpu + '-SL -U -C -DN  -c',  n + wide,  ''),
	('testing_ztrsm', ngpu + '-SL -U -C -DU  -c',  n + wide,  ''),
	
	('testing_ztrsm', ngpu + '-SR -L    -DN  -c',  n + tall,  ''),
	('testing_ztrsm', ngpu + '-SR -L    -DU  -c',  n + tall,  ''),
	('testing_ztrsm', ngpu + '-SR -L -C -DN  -c',  n + tall,  ''),
	('testing_ztrsm', ngpu + '-SR -L -C -DU  -c',  n + tall,  ''),
	
	('testing_ztrsm', ngpu + '-SR -U    -DN  -c',  n + tall,  ''),
	('testing_ztrsm', ngpu + '-SR -U    -DU  -c',  n + tall,  ''),
	('testing_ztrsm', ngpu + '-SR -U -C -DN  -c',  n + tall,  ''),
	('testing_ztrsm', ngpu + '-SR -U -C -DU  -c',  n + tall,  ''),
	
	# lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrsv',       '-L    -DN  -c',  n,    'cublas only'),
	('testing_ztrsv',       '-L    -DU  -c',  n,    'cublas only'),
	('testing_ztrsv',       '-L -C -DN  -c',  n,    'cublas only'),
	('testing_ztrsv',       '-L -C -DU  -c',  n,    'cublas only'),
	
	('testing_ztrsv',       '-U    -DN  -c',  n,    'cublas only'),
	('testing_ztrsv',       '-U    -DU  -c',  n,    'cublas only'),
	('testing_ztrsv',       '-U -C -DN  -c',  n,    'cublas only'),
	('testing_ztrsv',       '-U -C -DU  -c',  n,    'cublas only'),
	
	('#testing_zhemm_mgpu',  ngpu + '-L -c',  n,    'tester needs updating'),
	('#testing_zhemm_mgpu',  ngpu + '-U -c',  n,    'tester needs updating'),
	('testing_zhemv_mgpu',   ngpu + '-L -c',  n,    ''),
	('testing_zhemv_mgpu',   ngpu + '-U -c',  n,    ''),
	('testing_zher2k_mgpu',  ngpu + '-L -c',  n,    ''),
	('testing_zher2k_mgpu',  ngpu + '-U -c',  n,    ''),
	
	('#testing_blas_z',                '-c',  mnk,  'takes long time; cublas only'),
	('testing_cblas_z',                '-c',  n,    ''),
)
if ( opts.blas ):
	tests += blas

aux = (
	# ----------
	# auxiliary
	('testing_zgeadd',     '--version 1 -c',  mn,   ''),
	('testing_zgeadd',     '--version 2 -c',  mn,   ''),
	('testing_zlacpy',                 '-c',  mn,   ''),
	('testing_zlag2c',                 '-c',  mn,   ''),
	('testing_zlange',                 '-c',  mn,   ''),
	
	# lower/upper handled internally in one call
	('testing_zlanhe',                 '-c',  n,    ''),
	
	('testing_zlarfg',                 '-c',  n,    ''),
	('testing_zlascl',                 '-c',  mn,   ''),
	('testing_zlaset',                 '-c',  mn,   ''),
	('testing_zlaset_band',            '-c',  mn,   ''),
	('testing_zlat2c',                 '-c',  n,    ''),
	('testing_znan_inf',               '-c',  mn,   ''),
	('testing_zprint',                 '-c',  '-N 10 -N 5,100 -N 100,5',  ''),
	
	# lower/upper
	('testing_zsymmetrize',  '-L        -c',  n,    ''),
	('testing_zsymmetrize',  '-U        -c',  n,    ''),
	
	# lower/upper
	('testing_zsymmetrize_tiles',  '-L  -c',  n,    ''),
	('testing_zsymmetrize_tiles',  '-U  -c',  n,    ''),
	
	('testing_zswap',                  '-c',  n,    ''),
	('testing_ztranspose',             '-c',  mn,   ''),
	
	# lower/upper
	('testing_ztrtri_diag',         '-L -c',  n,    ''),
	('testing_ztrtri_diag',         '-U -c',  n,    ''),
	
	#('testing_auxiliary',             '-c',  '',   ''),  # run_tests misinterprets output as errors
	('testing_constants',              '-c',  '',   ''),
	('testing_operators',              '-c',  '',   ''),
	('testing_parse_opts',             '-c',  '',   ''),
)
if ( opts.aux ):
	tests += aux

chol = (
	# ----------
	# Cholesky, GPU interface
	('testing_zcposv_gpu',       '-L    -c',  n,    ''),
	('testing_zcposv_gpu',       '-U    -c',  n,    ''),
	
	('testing_zposv_gpu',        '-L    -c',  n,    ''),
	('testing_zposv_gpu',        '-U    -c',  n,    ''),
	
	('testing_zpotrf_gpu',       '-L   -c2',  n,    ''),
	('testing_zpotrf_gpu',       '-U   -c2',  n,    ''),
	
	('testing_zpotf2_gpu',       '-L    -c',  n + tall,  ''),
	('testing_zpotf2_gpu',       '-U    -c',  n + tall,  ''),
	
	('testing_zpotri_gpu',       '-L    -c',  n,    ''),
	('testing_zpotri_gpu',       '-U    -c',  n,    ''),
	
	# lower/upper, unit/non-unit
	('testing_ztrtri_gpu',      '-L -DU -c',  n,    ''),
	('testing_ztrtri_gpu',      '-L -DN -c',  n,    ''),
	('testing_ztrtri_gpu',      '-U -DU -c',  n,    ''),
	('testing_ztrtri_gpu',      '-U -DN -c',  n,    ''),
	
	('testing_zpotrf_mgpu', ngpu + '-L    -c',  n,    ''),
	('testing_zpotrf_mgpu', ngpu + '-U    -c',  n,    ''),
	
	# ----------
	# Cholesky, CPU interface
	('testing_zposv',            '-L    -c',  n,    ''),
	('testing_zposv',            '-U    -c',  n,    ''),
	
	('testing_zpotrf',           '-L    -c',  n,    ''),
	('testing_zpotrf',           '-U    -c',  n,    ''),
	
	('testing_zpotri',           '-L    -c',  n,    ''),
	('testing_zpotri',           '-U    -c',  n,    ''),
	
	# lower/upper, unit/non-unit
	('testing_ztrtri',          '-L -DU -c',  n,    ''),
	('testing_ztrtri',          '-L -DN -c',  n,    ''),
	('testing_ztrtri',          '-U -DU -c',  n,    ''),
	('testing_ztrtri',          '-U -DN -c',  n,    ''),
)
if ( opts.chol ):
	tests += chol

hesv = (
	# ----------
	# Symmetric Indefinite
	('testing_zhesv',               '-L -c',  n,    ''),
	('testing_zhesv',               '-U -c',  n,    ''),
	
	('testing_zhesv_nopiv_gpu',     '-L -c',  n,    ''),
	('testing_zhesv_nopiv_gpu',     '-U -c',  n,    ''),
	
	('testing_zsysv_nopiv_gpu',     '-L -c',  n,    ''),
	('testing_zsysv_nopiv_gpu',     '-U -c',  n,    ''),
	
	# Bunch-Kauffman
	('testing_zhetrf', '-L --version 1 -c2',  n,    ''),
	('testing_zhetrf', '-U --version 1 -c2',  n,    ''),
	
	# no-pivot LDLt, CPU interface
	('testing_zhetrf', '-L --version 3 -c2',  n,    ''),
	('testing_zhetrf', '-U --version 3 -c2',  n,    ''),
	
	# no-pivot LDLt, GPU interface
	('testing_zhetrf', '-L --version 4 -c2',  n,    ''),
	('testing_zhetrf', '-U --version 4 -c2',  n,    ''),
	
	# Aasen's
	('testing_zhetrf', '-L --version 6 -c2',  n,    ''),
	('#testing_zhetrf','-U --version 6 -c2',  n,    'upper not implemented'),
)
if ( opts.hesv ):
	tests += hesv

lu = (
	# ----------
	# LU, GPU interface
	('testing_zcgesv_gpu',             '-c',  n,    ''),
	('testing_zgesv_gpu',              '-c',  n,    ''),
	('testing_zgetrf_gpu', '--version 1 -c2', n,    ''),
	('testing_zgetrf_gpu', '--version 2 -c2', n,    ''), # zgetrf_nopiv_gpu
	('testing_zgetf2_gpu',             '-c',  n + tall,  ''),
	('testing_zgetri_gpu',             '-c',  n,    ''),
	('testing_zgetrf_mgpu',    ngpu + '-c2',  n,    ''),
	
	# ----------
	# LU, CPU interface
	('testing_zgesv',                  '-c',  n,    ''),
	('testing_zgesv_rbt',              '-c',  n,    ''),
	('testing_zgetrf',    '--version 1 -c2',  n,    ''),
	('testing_zgetrf',    '--version 2 -c2',  n,    ''),  # zgetrf_nopiv
	('testing_zgetrf',    '--version 3 -c2',  n,    ''),  # zgetf2_nopiv
)
if ( opts.lu ):
	tests += lu

qr = (
	# ----------
	# QR and least squares, GPU interface
	('testing_zcgeqrsv_gpu',           '-c',  mn,   ''),
	
	('testing_zgegqr_gpu', '--version 1 -c',  mn,   ''),
	('testing_zgegqr_gpu', '--version 2 -c',  mn,   ''),
	('testing_zgegqr_gpu', '--version 3 -c',  mn,   ''),
	('testing_zgegqr_gpu', '--version 4 -c',  mn,   ''),
	
	('testing_zgelqf_gpu',             '-c',  mn,   ''),
	('testing_zgels_gpu',              '-c',  mn,   ''),
	('testing_zgels3_gpu',             '-c',  mn,   ''),
	
	('testing_zgeqp3_gpu',             '-c',  mn,   ''),
	('testing_zgeqr2_gpu',             '-c',  mn,   ''),
	
	('testing_zgeqr2x_gpu', '--version 1 -c', mn,   ''),
	('testing_zgeqr2x_gpu', '--version 2 -c', mn,   ''),
	('testing_zgeqr2x_gpu', '--version 3 -c', mn,   ''),
	('testing_zgeqr2x_gpu', '--version 4 -c', mn,   ''),
	
	('testing_zgeqrf_gpu', '--version 1 -c2', mn,   ''),
	('testing_zgeqrf_gpu', '--version 2 -c2', mn,   ''),
	('testing_zgeqrf_gpu', '--version 3 -c2', mn,   ''),
	
	('testing_zlarfb_gpu', '--version 1 -c',  mnk,  ''),
	('testing_zlarfb_gpu', '--version 2 -c',  mnk,  ''),
	('testing_zungqr_gpu',             '-c',  mnk,  ''),
	('testing_zunmql_gpu',             '-c',  mnk,  ''),
	('testing_zunmqr_gpu', '--version 1 -c',  mnk,  ''),
	('testing_zunmqr_gpu', '--version 2 -c',  mnk,  ''),
	('testing_zgeqrf_mgpu',    ngpu + '-c2',  mn,   ''),
	
	# ----------
	# QR, CPU interface
	('testing_zgelqf',                 '-c',  mn,   ''),
	('testing_zgels',                  '-c',  mn,   ''),
	('testing_zgeqlf',                 '-c',  mn,   ''),
	('testing_zgeqp3',                 '-c',  mn,   ''),
	('testing_zgeqrf',                '-c2',  mn,   ''),
	('testing_zunglq',                 '-c',  mnk,  ''),
	('testing_zungqr',     '--version 1 -c',  mnk,  ''),
	('testing_zungqr',     '--version 2 -c',  mnk,  ''),
	('testing_zungqr',          ngpu + '-c',  mnk,  ''),
	('testing_zunmlq',                 '-c',  mnk,  ''),
	('testing_zunmql',                 '-c',  mnk,  ''),
	('testing_zunmqr',                 '-c',  mnk,  ''),
)
if ( opts.qr ):
	tests += qr

syev = (
	# ----------
	# symmetric eigenvalues, GPU interface
	# no-vectors/vectors, lower/upper
	('testing_zheevd_gpu',      '--version 1 -L -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 1 -U -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 1 -L -JV -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 1 -U -JV -c',  n,    ''),
	
	# version 2 is zheevdx_gpu
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevd_gpu',      '--version 2 --fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 2 --fraction 1.0 -U -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 2 --fraction 1.0 -L -JV -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 2 --fraction 1.0 -U -JV -c',  n,    ''),
	
	# version 3 is zheevr_gpu
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevd_gpu',      '--version 3 --fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 3 --fraction 1.0 -U -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 3 --fraction 1.0 -L -JV -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 3 --fraction 1.0 -U -JV -c',  n,    ''),
	
	# version 4 is zheevx_gpu
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevd_gpu',      '--version 4 --fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 4 --fraction 1.0 -U -JN -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 4 --fraction 1.0 -L -JV -c',  n,    ''),
	('testing_zheevd_gpu',      '--version 4 --fraction 1.0 -U -JV -c',  n,    ''),
	
	# lower/upper, version 1 (cublas_hemv)/2 (fast_hemv)
	('testing_zhetrd_gpu',  '--version 1 -L -c',  n,    ''),
	('testing_zhetrd_gpu',  '--version 1 -U -c',  n,    ''),
	('testing_zhetrd_gpu',  '--version 2 -L -c',  n,    ''),
	('testing_zhetrd_gpu',  '--version 2 -U -c',  n,    ''),
	
	# multi-gpu
	('testing_zhetrd_mgpu', ngpu + '-L     -c',  n,    ''),
	('testing_zhetrd_mgpu', ngpu + '-U     -c',  n,    ''),
	
	# ----------
	# symmetric eigenvalues, CPU interface
	# no vectors/vectors, lower/upper
	('testing_zheevd', '--version 1 -L -JN -c',  n,    ''),
	('testing_zheevd', '--version 1 -U -JN -c',  n,    ''),
	('testing_zheevd', '--version 1 -L -JV -c',  n,    ''),
	('testing_zheevd', '--version 1 -U -JV -c',  n,    ''),
	
	# version 2 is zheevdx
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevd', '--version 2 --fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevd', '--version 2 --fraction 1.0 -U -JN -c',  n,    ''),
	('testing_zheevd', '--version 2 --fraction 1.0 -L -JV -c',  n,    ''),
	('testing_zheevd', '--version 2 --fraction 1.0 -U -JV -c',  n,    ''),
	
	# version 3 is zheevr
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevd', '--version 3 --fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevd', '--version 3 --fraction 1.0 -U -JN -c',  n,    ''),
	('testing_zheevd', '--version 3 --fraction 1.0 -L -JV -c',  n,    ''),
	('testing_zheevd', '--version 3 --fraction 1.0 -U -JV -c',  n,    ''),
	
	# version 4 is zheevx
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevd', '--version 4 --fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevd', '--version 4 --fraction 1.0 -U -JN -c',  n,    ''),
	('testing_zheevd', '--version 4 --fraction 1.0 -L -JV -c',  n,    ''),
	('testing_zheevd', '--version 4 --fraction 1.0 -U -JV -c',  n,    ''),
	
	# lower/upper
	('testing_zhetrd',          '-L     -c',  n,    ''),
	('testing_zhetrd',          '-U     -c',  n,    ''),
	
	# ----------
	# symmetric eigenvalues, 2-stage
	#('testing_zhetrd_he2hb',       '-L -c',  n,    'NOT hetrd_he2hb -- calls heevdx_2stage'),
	#('testing_zhetrd_he2hb',       '-U -c',  n,    'NOT hetrd_he2hb -- calls heevdx_2stage. upper not implemented'),
	
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevdx_2stage',  '--fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevdx_2stage',  '--fraction 1.0 -L -JV -c',  n,    ''),
	('#testing_zheevdx_2stage', '--fraction 1.0 -U -JN -c',  n,    'upper not implemented'),
	('#testing_zheevdx_2stage', '--fraction 1.0 -U -JV -c',  n,    'upper not implemented'),
	
	# same tester for multi-GPU version
	# TODO test multi-GPU version with ngpu=1
	# TODO test with --fraction < 1; checks don't seem to work.
	('testing_zheevdx_2stage',  ngpu + '--fraction 1.0 -L -JN -c',  n,    ''),
	('testing_zheevdx_2stage',  ngpu + '--fraction 1.0 -L -JV -c',  n,    ''),
	('#testing_zheevdx_2stage', ngpu + '--fraction 1.0 -U -JN -c',  n,    'upper not implemented'),
	('#testing_zheevdx_2stage', ngpu + '--fraction 1.0 -U -JV -c',  n,    'upper not implemented'),
)
if ( opts.syev ):
	tests += syev

sygv = (
	# ----------
	# generalized symmetric eigenvalues
	('testing_zhegst',               '-L --itype 1 -c',  n,  ''),
	('testing_zhegst',               '-L --itype 2 -c',  n,  ''),
	# itype 2 == itype 3 for hegst
	
	('testing_zhegst',               '-U --itype 1 -c',  n,  ''),
	('testing_zhegst',               '-U --itype 2 -c',  n,  ''),
	
	('testing_zhegst_gpu',           '-L --itype 1 -c',  n,  ''),
	('testing_zhegst_gpu',           '-L --itype 2 -c',  n,  ''),
	
	('testing_zhegst_gpu',           '-U --itype 1 -c',  n,  ''),
	('testing_zhegst_gpu',           '-U --itype 2 -c',  n,  ''),
	
	# no-vector/vector, lower/upper, itypes
	('testing_zhegvd',           '-L -JN --itype 1 -c',  n,  ''),
	('testing_zhegvd',           '-L -JN --itype 2 -c',  n,  ''),
	('testing_zhegvd',           '-L -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvd',           '-U -JN --itype 1 -c',  n,  ''),
	('testing_zhegvd',           '-U -JN --itype 2 -c',  n,  ''),
	('testing_zhegvd',           '-U -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvd',           '-L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvd',           '-L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvd',           '-L -JV --itype 3 -c',  n,  ''),
	
	('testing_zhegvd',           '-U -JV --itype 1 -c',  n,  ''),
	('testing_zhegvd',           '-U -JV --itype 2 -c',  n,  ''),
	('testing_zhegvd',           '-U -JV --itype 3 -c',  n,  ''),
	
	# lower/upper, no-vector/vector, itypes, add ngpu to call zhegvd_m
	('testing_zhegvd',    ngpu + '-L -JN --itype 1 -c',  n,  ''),
	('testing_zhegvd',    ngpu + '-L -JN --itype 2 -c',  n,  ''),
	('testing_zhegvd',    ngpu + '-L -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvd',    ngpu + '-U -JN --itype 1 -c',  n,  ''),
	('testing_zhegvd',    ngpu + '-U -JN --itype 2 -c',  n,  ''),
	('testing_zhegvd',    ngpu + '-U -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvd',    ngpu + '-L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvd',    ngpu + '-L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvd',    ngpu + '-L -JV --itype 3 -c',  n,  ''),
	
	('testing_zhegvd',    ngpu + '-U -JV --itype 1 -c',  n,  'upper not implemented ??'),
	('testing_zhegvd',    ngpu + '-U -JV --itype 2 -c',  n,  'upper not implemented ??'),
	('testing_zhegvd',    ngpu + '-U -JV --itype 3 -c',  n,  'upper not implemented ??'),
	
	# lower/upper, no-vector/vector, itypes
	# TODO fraction
	('testing_zhegvdx',          '--version 1 -L -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -L -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -L -JN --itype 3 -c',  n,  ''),
	 
	('testing_zhegvdx',          '--version 1 -U -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -U -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -U -JN --itype 3 -c',  n,  ''),
	 
	('testing_zhegvdx',          '--version 1 -L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -L -JV --itype 3 -c',  n,  ''),
	 
	('testing_zhegvdx',          '--version 1 -U -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -U -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx',          '--version 1 -U -JV --itype 3 -c',  n,  ''),
	
	# lower/upper, no-vector/vector, itypes, add ngpu to call zhegvdx_m
	# TODO fraction
	('testing_zhegvdx',   ngpu + '--version 1 -L -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -L -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -L -JN --itype 3 -c',  n,  ''),
	 
	('testing_zhegvdx',   ngpu + '--version 1 -U -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -U -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -U -JN --itype 3 -c',  n,  ''),
	 
	('testing_zhegvdx',   ngpu + '--version 1 -L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -L -JV --itype 3 -c',  n,  ''),
	 
	('testing_zhegvdx',   ngpu + '--version 1 -U -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -U -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx',   ngpu + '--version 1 -U -JV --itype 3 -c',  n,  ''),
	
	# version 2 is zhegvx
	# TODO fraction
	('testing_zhegvdx', '--version 2 -L -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -L -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -L -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvdx', '--version 2 -U -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -U -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -U -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvdx', '--version 2 -L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -L -JV --itype 3 -c',  n,  ''),
	
	('testing_zhegvdx', '--version 2 -U -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -U -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 2 -U -JV --itype 3 -c',  n,  ''),
	
	# version 2 is zhegvr
	# TODO fraction
	('testing_zhegvdx', '--version 3 -L -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -L -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -L -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvdx', '--version 3 -U -JN --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -U -JN --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -U -JN --itype 3 -c',  n,  ''),
	
	('testing_zhegvdx', '--version 3 -L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -L -JV --itype 3 -c',  n,  ''),
	
	('testing_zhegvdx', '--version 3 -U -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -U -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx', '--version 3 -U -JV --itype 3 -c',  n,  ''),
	
	# lower/upper, no-vector/vector, itypes
	# TODO: add -c
	('testing_zhegvdx_2stage',   '-L -JN --itype 1',  n,  '-c implies -JV'),
	('testing_zhegvdx_2stage',   '-L -JN --itype 2',  n,  '-c implies -JV'),
	('testing_zhegvdx_2stage',   '-L -JN --itype 3',  n,  '-c implies -JV'),
	
	('#testing_zhegvdx_2stage',  '-U -JN --itype 1',  n,  '-c implies -JV, upper not implemented'),
	('#testing_zhegvdx_2stage',  '-U -JN --itype 2',  n,  '-c implies -JV, upper not implemented'),
	('#testing_zhegvdx_2stage',  '-U -JN --itype 3',  n,  '-c implies -JV, upper not implemented'),
	
	('testing_zhegvdx_2stage',   '-L -JV --itype 1 -c',  n,  ''),
	('testing_zhegvdx_2stage',   '-L -JV --itype 2 -c',  n,  ''),
	('testing_zhegvdx_2stage',   '-L -JV --itype 3 -c',  n,  ''),
	
	('#testing_zhegvdx_2stage',  '-U -JV --itype 1 -c',  n,  'upper not implemented'),
	('#testing_zhegvdx_2stage',  '-U -JV --itype 2 -c',  n,  'upper not implemented'),
	('#testing_zhegvdx_2stage',  '-U -JV --itype 3 -c',  n,  'upper not implemented'),
	
	# lower/upper, no-vector/vector, itypes
	# TODO: add -c
	('testing_zhegvdx_2stage',  ngpu + '-L -JN --itype 1', n,  '-c implies -JV'),
	('testing_zhegvdx_2stage',  ngpu + '-L -JN --itype 2', n,  '-c implies -JV'),
	('testing_zhegvdx_2stage',  ngpu + '-L -JN --itype 3', n,  '-c implies -JV'),
	
	('#testing_zhegvdx_2stage', ngpu + '-U -JN --itype 1', n,  '-c implies -JV, upper not implemented'),
	('#testing_zhegvdx_2stage', ngpu + '-U -JN --itype 2', n,  '-c implies -JV, upper not implemented'),
	('#testing_zhegvdx_2stage', ngpu + '-U -JN --itype 3', n,  '-c implies -JV, upper not implemented'),
	
	('testing_zhegvdx_2stage',  ngpu + '-L -JV --itype 1 -c', n,  ''),
	('testing_zhegvdx_2stage',  ngpu + '-L -JV --itype 2 -c', n,  ''),
	('testing_zhegvdx_2stage',  ngpu + '-L -JV --itype 3 -c', n,  ''),
	
	('#testing_zhegvdx_2stage', ngpu + '-U -JV --itype 1 -c', n,  'upper not implemented'),
	('#testing_zhegvdx_2stage', ngpu + '-U -JV --itype 2 -c', n,  'upper not implemented'),
	('#testing_zhegvdx_2stage', ngpu + '-U -JV --itype 3 -c', n,  'upper not implemented'),
)
if ( opts.sygv ):
	tests += sygv

geev = (
	# ----------
	# non-symmetric eigenvalues
	# right & left no-vector/vector; not all combos are tested here
	#('testing_dgeev',                   '',  n,    ''),  # covered by testing_zgeev
	('testing_zgeev',          '-RN -LN -c',  n,    ''),
	('testing_zgeev',          '-RV -LV -c',  n,    ''),
	('testing_zgeev',   ngpu + '-RN -LN -c',  n,    ''),
	('testing_zgeev',   ngpu + '-RV -LV -c',  n,    ''),
	
	('testing_zgehrd',     '--version 1 -c',  n,    ''),
	('testing_zgehrd',     '--version 2 -c',  n,    ''),
	('testing_zgehrd',          ngpu + '-c',  n,    ''),
)
if ( opts.geev ):
	tests += geev

svd = (
	# ----------
	# SVD
	# U & V none/some/overwrite/all
	# gesdd only has one jobz (taken from -U), while
	# gesvd can set U & V independently; not all combos are tested here
	('testing_zgesdd',         '-UN     -c',  mn,   ''),
	('testing_zgesdd',         '-US     -c',  mn,   ''),
	('testing_zgesdd',         '-UO     -c',  mn,   ''),
	('testing_zgesdd',         '-UA     -c',  n,    ''),  # todo: do tall & wide, but avoid excessive sizes
	
	('testing_zgesvd',         '-UN -VN -c',  mn,   ''),
	('testing_zgesvd',         '-US -VS -c',  mn,   ''),
	('testing_zgesvd',         '-UO -VS -c',  mn,   ''),
	('testing_zgesvd',         '-UA -VA -c',  n,    ''),  # todo: do tall & wide, but avoid excessive sizes
	
	('testing_zgebrd',                 '-c',  mn,   ''),
	('testing_zungbr',                 '-c',  mnk,  ''),
	('testing_zunmbr',                 '-c',  mnk,  ''),
)
if ( opts.svd ):
	tests += svd

batched = (
	# ----------
	# batched (BLAS, LU, etc.)
	('testing_zgeadd_batched',    batch + '               -c',  mn,   ''),
	
	# no-trans/conj-trans; there are other combinations with trans
	('testing_zgemm_batched',     batch + '-NN            -c',  mn,   ''),
	('testing_zgemm_batched',     batch + '-NC            -c',  mn,   ''),
	('testing_zgemm_batched',     batch + '-CN            -c',  mn,   ''),
	('testing_zgemm_batched',     batch + '-CC            -c',  mn,   ''),
	
	# no-trans/trans/conj-trans
	('testing_zgemv_batched',     batch + '               -c',  mn,   ''),
	('testing_zgemv_batched',     batch + '-T             -c',  mn,   ''),
	('testing_zgemv_batched',     batch + '-C             -c',  mn,   ''),
	
	# lower/upper, no-trans/conj-trans
	('testing_zherk_batched',     batch + '         -L    -c',  mn,   ''),
	('testing_zherk_batched',     batch + '         -L -C -c',  mn,   ''),
	('testing_zherk_batched',     batch + '         -U    -c',  mn,   ''),
	('testing_zherk_batched',     batch + '         -U -C -c',  mn,   ''),
	
	('testing_zlacpy_batched',    batch + '               -c',  mn,   ''),
	
	# left/right, lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrsm_batched',     batch + '-SL -L    -DN  -c',  n + wide, ''),
	('testing_ztrsm_batched',     batch + '-SL -L    -DU  -c',  n + wide, ''),
	('testing_ztrsm_batched',     batch + '-SL -L -C -DN  -c',  n + wide, ''),
	('testing_ztrsm_batched',     batch + '-SL -L -C -DU  -c',  n + wide, ''),
	
	('testing_ztrsm_batched',     batch + '-SL -U    -DN  -c',  n + wide, ''),
	('testing_ztrsm_batched',     batch + '-SL -U    -DU  -c',  n + wide, ''),
	('testing_ztrsm_batched',     batch + '-SL -U -C -DN  -c',  n + wide, ''),
	('testing_ztrsm_batched',     batch + '-SL -U -C -DU  -c',  n + wide, ''),
	
	('testing_ztrsm_batched',     batch + '-SR -L    -DN  -c',  n + tall, ''),
	('testing_ztrsm_batched',     batch + '-SR -L    -DU  -c',  n + tall, ''),
	('testing_ztrsm_batched',     batch + '-SR -L -C -DN  -c',  n + tall, ''),
	('testing_ztrsm_batched',     batch + '-SR -L -C -DU  -c',  n + tall, ''),
	
	('testing_ztrsm_batched',     batch + '-SR -U    -DN  -c',  n + tall, ''),
	('testing_ztrsm_batched',     batch + '-SR -U    -DU  -c',  n + tall, ''),
	('testing_ztrsm_batched',     batch + '-SR -U -C -DN  -c',  n + tall, ''),
	('testing_ztrsm_batched',     batch + '-SR -U -C -DU  -c',  n + tall, ''),
	
	# lower/upper, no-trans/conj-trans, non-unit/unit diag
	('testing_ztrsv_batched',     batch + '    -L    -DN  -c',  n,    ''),
	('testing_ztrsv_batched',     batch + '    -L    -DU  -c',  n,    ''),
	('testing_ztrsv_batched',     batch + '    -L -C -DN  -c',  n,    ''),
	('testing_ztrsv_batched',     batch + '    -L -C -DU  -c',  n,    ''),
	
	('testing_ztrsv_batched',     batch + '    -U    -DN  -c',  n,    ''),
	('testing_ztrsv_batched',     batch + '    -U    -DU  -c',  n,    ''),
	('testing_ztrsv_batched',     batch + '    -U -C -DN  -c',  n,    ''),
	('testing_ztrsv_batched',     batch + '    -U -C -DU  -c',  n,    ''),
	
	# ----- QR
	('testing_zgeqrf_batched',    batch + '               -c',  mn,   ''),
	
	# ----- LU
	('testing_zgesv_batched',         batch + '           -c',  mn,   ''),
	('testing_zgesv_nopiv_batched',   batch + '           -c',  mn,   ''),
	('testing_zgetrf_batched',        batch + '          -c2',  mn,   ''),
	('testing_zgetrf_nopiv_batched',  batch + '          -c2',  mn,   ''),
	('testing_zgetri_batched',        batch + '           -c',  n,    ''),
	
	# ----- Cholesky
	('testing_zposv_batched',     batch + '         -L    -c',  n,    ''),
	('#testing_zposv_batched',    batch + '         -U    -c',  n,    'upper not implemented'),
	
	('testing_zpotrf_batched',    batch + '         -L    -c2', n,    ''),
	('#testing_zpotrf_batched',   batch + '         -U    -c2', n,    'upper not implemented'),
)
if ( opts.batched ):
	tests += batched


# ----------------------------------------------------------------------
# multi-GPU (BLAS, LU, etc.) -- take from other sets
mgpu = []
for s in (blas, aux, chol, hesv, lu, qr, syev, sygv, geev, svd):
	for test in s:
		m1 = re.search( '(_m|_mgpu)$', test[0] )
		m2 = re.search( '--ngpu',      test[1] )
		if (m1 or m2):
			mgpu.append( test )
	# end
# end
if ( opts.mgpu ):
	tests += mgpu


# ----------------------------------------------------------------------
if ( opts.no_mgpu ):
	tests2 = []
	for test in tests:
		m1 = re.search( '(_m|_mgpu)$', test[0] )
		m2 = re.search( '--ngpu',      test[1] )
		if ( not (m1 or m2) ):
			tests2.append( test )
	# end
	tests = tests2
# end

# ----------------------------------------------------------------------
# select subset of commands
options = []
if (opts.itype):
	options.append('--itype %s' % (opts.itype))
if (opts.version):
	options.append('--version %s' % (opts.version))
if (opts.lower):
	options.append('-L')
elif (opts.upper):
	options.append('-U')
if (opts.jobz):
	options.append('-J%s' % (opts.jobz))
if (opts.diag):
	options.append('-D%s' % (opts.diag))
if (opts.fraction):
	options.append('--fraction %s' % (opts.fraction))

if len(options) > 0:
	tests2 = []
	for test in tests:
		match = True
		for o in options:
			if not re.search( o, test[1] ):
				match = False
				break
		if match:
			tests2.append( test )
	# end
	tests = tests2
# end


# ----------------------------------------------------------------------
precisions = (
	's', 'd', 'c', 'z'
)

subs = (
	('',              'testing_dlag2s', '',              'testing_zlag2c'),
	('',              'testing_dlat2s', '',              'testing_zlat2c'),
	('ssy',           'dsy',            'che',           'zhe'           ),
	('sor',           'dor',            'cun',           'zun'           ),
	('sy2sb',         'sy2sb',          'he2hb',         'he2hb'         ),
	('',              'testing_ds',     '',              'testing_zc'    ),
	('testing_s',     'testing_d',      'testing_c',     'testing_z'     ),
	('lansy',         'lansy',          'lanhe',         'lanhe'         ),
	('blas_s',        'blas_d',         'blas_c',        'blas_z'        ),
)

# ----------
# simple precision generation
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
# runs command in a subprocess.
# returns list (okay, fail, errors, status)
# okay   is count of "ok"     in output.
# fail   is count of "failed" in output.
# error  is count of indications of other errors (exit, CUDA error, etc.).
# status is exit status of the command.
def run( cmd ):
	words = re.split( ' +', cmd.strip() )
	
	# stdout & stderr are merged
	p = subprocess.Popen( words, bufsize=1, stdout=PIPE, stderr=STDOUT )
	
	okay  = 0
	fail  = 0
	error = 0
	# read unbuffered ("for line in p.stdout" will buffer)
	while True:
		line = p.stdout.readline()
		if not line:
			break
		print line.rstrip()
		if re.search( r'\bok *$', line ):
			okay += 1
		if re.search( 'failed', line ):
			fail += 1
		if re.search( 'exit|memory leak|memory mapping error|CUDA runtime error|CL_INVALID|illegal value|ERROR SUMMARY: [1-9]', line ):
			error += 1
	# end
	
	status = p.wait()
	return (okay, fail, error, status)
# end


# ----------------------------------------------------------------------
ntest  = 0
nokay  = 0
nfail  = 0
nerror = 0
failures = {}

start = None
if ( opts.start ):
	start = re.compile( opts.start + r'\b' )

seen  = {}
pause = 0

global_options = ''
if ( opts.tol ):
	global_options += ' --tol ' + opts.tol + ' '

if ( opts.dev is not None ):
	global_options += ' --dev ' + opts.dev + ' '

if ( opts.null_stream ):
	global_options += ' --null-stream '

if ( opts.niter != 1 ):
	global_options += ' --niter ' + opts.niter + ' '

last_cmd = None

for test in tests:
	(cmd, options, sizes, comments) = test
	
	make = False
	for precision in opts.precisions:
		# precision generation
		# in a few cases this doesn't produce a valid tester name (e.g., testing_zcposv_gpu -> posv_gpu)
		cmdp = substitute( cmd, 'z', precision )
		if ( not re.match( 'testing_', cmdp )):
			continue
		
		disabled = (cmdp[0] == '#')
		if ( disabled ):
			cmdp = cmdp[1:]
		
		# command to run
		cmd_args = './' + cmdp +' '+ options +' '+ global_options + sizes
		cmd_args = re.sub( '  +', ' ', cmd_args )  # compress spaces
		
		# command to print on console, lacks sizes
		cmd_opts = cmdp +' '+ options
		cmd_opts = re.sub( '  +', ' ', cmd_opts )  # compress spaces
				
		# skip tests before start
		if ( start and not start.search( cmdp )):
			continue
		start = None
		
		# skip tests not in args, or duplicates
		# skip and warn about non-existing
		if (    (args and not cmdp in args)
		     or (seen.has_key( cmd_opts )) ):
			continue
		# end
		if ( not os.path.exists( cmdp )):
			print >>sys.stderr, cmdp, "doesn't exist (original name: " + cmd + ", precision: " + precision + ")"
			continue
		# end
		seen[ cmd_opts ] = True
		
		if ( opts.memcheck ):
			cmd_args = 'cuda-memcheck ' + cmd_args
		
		go = True
		while( go ):
			if pause > 0:
				time.sleep( pause )
				pause = 0
			# end
			
			print
			print '*'*100
			print cmd_args
			print '*'*100
			sys.stdout.flush()
			
			if ( non_interactive ):
				if ( last_cmd and cmd != last_cmd ):
					sys.stderr.write( '\n' )
				last_cmd = cmd
				sys.stderr.write( '%-40s' % cmd_opts )
				sys.stderr.flush()
			# end
			
			if ( disabled ):
				if ( comments ):
					sys.stderr.write( '  (disabled: ' + comments + ')\n' )
				else:
					sys.stderr.write( '  (disabled)\n' )
				go = False
				continue
			# end
			
			if ( make ):
				m = 'make lib ' + cmdp
				print m
				run( m )
			# end
			
			t = time.time()
			(okay, fail, error, status) = run( cmd_args )
			t = time.time() - t
			
			# count stats
			ntest  += 1
			nokay  += okay
			nfail  += fail
			nerror += error
			
			errmsg = ''
			if ( fail > 0 ):
				errmsg += '  ** %d tests failed' % (fail)
			if ( error > 0 ):
				errmsg += '  ** %d errors' % (error)
			if ( status < 0 ):
				errmsg += '  ** exit with signal %d' % (-status)
				nerror += 1  # count crash as an error
			
			if ( errmsg != '' ):
				if ( non_interactive ):
					sys.stderr.write( errmsg + '\n' )  # to console
				sys.stdout.write( errmsg + '\n' )  # to file
				failures[ cmd_opts ] = True
			else:
				sys.stderr.write( '  ok\n' )
			# end
			
			if ( non_interactive ):
				# set to sleep a few seconds before next test,
				# to allow processor to cool off some between tests.
				pause = min( t, 5. )
				go = False
			else:
				x = raw_input( '[enter to continue; M to make and re-run] ' )
				if ( x in ('m','M')):
					make = True
				else:
					go = False
			# endif
		# end
	# end
# end


# print summary
msg  = '\n'
msg += '*'*100   + '\n'
msg += 'summary' + '\n'
msg += '*'*100   + '\n'

if ( nfail == 0 and nerror == 0 ):
	msg += 'all %d tests in %d commands passed!\n' % (nokay, ntest)
else:
	msg += '%5d tests in %d commands passed\n' % (nokay, ntest)
	msg += '%5d tests failed accuracy test\n' % (nfail)
	msg += '%5d errors detected (crashes, CUDA errors, etc.)\n' % (nerror)
	f = failures.keys()
	f.sort()
	msg += 'routines with failures:\n    ' + '\n    '.join( f ) + '\n'
# end

if ( non_interactive ):
	sys.stderr.write( msg )  # to console
sys.stdout.write( msg )  # to file
