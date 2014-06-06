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
# Typing "m" re-makes and re-runs that tester.
# Typing enter goes to the next tester.
#
# The --start option skips all testers before the given one, then continues
# with testers from there.
#   ./runs.py --start testing_spotrf
#
# If specific testers are named on the command line, only those are run.
#   ./runs.py testing_spotrf testing_sgetrf
#
# The -p option controls what precisions are tested, the default being "sdcz"
# for all precisions.
#   ./runs.py -p sd
#
# The -s -m -l options control what sizes are tested, the default being
# all three sets.
#   ./runs.py -s -m
#
# @author Mark Gates

import os
import re
import sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-p', '--precisions', action='store',      dest='precisions', help='run given precisions (initials, e.g., "sd" for single and double)', default='sdcz')
parser.add_option(      '--start',      action='store',      dest='start',      help='start with given routine')
parser.add_option('-s', '--small',      action='store_true', dest='small',      help='run small  tests, N <= 300')
parser.add_option('-m', '--medium',     action='store_true', dest='med',        help='run medium tests, 300 <= N <= 2000')
parser.add_option('-l', '--large',      action='store_true', dest='large',      help='run large  tests, 2000 <= N')
(opts, args) = parser.parse_args()

if ( not opts.small and not opts.med and not opts.large ):
	opts.small = True
	opts.med   = True
	opts.large = True
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
	n +=  ' -N 510  -N 511  -N 512  -N 513  -N 514 --range 100:400:100'
if opts.large:
	n +=  ' --range 1000:4000:1000'


# ----------
tall = ''
if opts.small:
	tall += (' -N 2,1        -N 3,1        -N 4,2'
	     +   ' -N 20,18      -N 20,10      -N 20,2      -N 20,1'
	     +   ' -N 200,180    -N 200,100    -N 200,20    -N 200,10    -N 200,1'
	)
if opts.med:
	tall +=  ' -N 600,180    -N 600,100    -N 600,20    -N 600,10    -N 600,1'
if opts.large:
	tall +=  ' -N 2000,1800  -N 2000,1000  -N 2000,200  -N 2000,100  -N 2000,10  -N 2000,1'


# ----------
wide = ''
if opts.small:
	wide += (' -N 1,2        -N 1,3        -N 2,4'
	     +   ' -N 18,20      -N 10,20      -N 2,20      -N 1,20'
	     +   ' -N 180,200    -N 100,200    -N 20,200    -N 10,200    -N 1,200'
	)
if opts.med:
	wide +=  ' -N 180,600    -N 100,600    -N 20,600    -N 10,600    -N 1,600'
if opts.large:
	wide +=  ' -N 1800,2000  -N 1000,2000  -N 200,2000  -N 100,2000  -N 10,2000  -N 1,2000'


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

tests = (
# ----------
# BLAS and auxiliary
	'./testing_z_cublas_v2       -c' + n,
	                       
	'./testing_zgemm             -c' + mnk,  # trans=N,N
	'./testing_zgemv             -c' + mn,
	'./testing_zhemv             -c' + n,
	'./testing_zherk             -c' + n,
	'./testing_zher2k            -c' + n,
	'./testing_zsymv             -c' + n,
	'./testing_ztrmm             -c' + mn,
	'./testing_ztrmv             -c' + n,
	'./testing_ztrsm             -c' + mn,
	'./testing_ztrsv             -c' + n,
	
	'./testing_auxiliary           ',
	'./testing_constants           ',
	'./testing_parse_opts          ',
	'./testing_zblas               ' + mnk,
	'./testing_zgeadd              ' + mn,
	'./testing_zgeadd_batched      ' + mn,
	'./testing_zlacpy              ' + mn,
	'./testing_zlacpy_batched      ' + mn,
	'./testing_zlag2c              ' + mn,
	'./testing_zlange              ' + mn,
	'./testing_zlanhe              ' + n,
	'./testing_zlarfg              ' + n,
	'./testing_zsymmetrize         ' + n,
	'./testing_zsymmetrize_tiles   ' + n,
	'./testing_zswap               ' + n,
	'./testing_ztranspose          ' + mn,
	
# ----------
# Cholesky, GPU interface
	'./testing_zcposv_gpu        -c' + n,
	'./testing_zposv_gpu         -c' + n,
	'./testing_zpotrf_gpu       -c2' + n,
	'./testing_zpotf2_gpu        -c' + n + tall,
	'./testing_zpotri_gpu        -c' + n,
	'./testing_zpotrf_mgpu       -c' + n,
	
# ----------
# Cholesky, CPU interface
	'./testing_zposv             -c' + n,
	'./testing_zpotrf            -c' + n,
	'./testing_zpotri            -c' + n,
	
# ----------
# LU, GPU interface
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
	
# ----------
# QR and least squares, GPU interface
# TODO qrf uses  -c2 ?
	'./testing_zcgeqrsv_gpu      -c' + mn,
	'./testing_zgelqf_gpu        -c' + mn,
	'./testing_zgels_gpu         -c' + mn,
	'./testing_zgels3_gpu        -c' + mn,
	'./testing_zgeqr2_gpu        -c' + mn,
	'./testing_zgeqr2x_gpu       -c' + mn,
	'./testing_zgeqrf_gpu        -c' + mn,
	'./testing_zlarfb_gpu        -c' + mnk,
	'./testing_zungqr_gpu        -c' + mnk,
	'./testing_zunmqr_gpu        -c' + mnk,
	'./testing_zgeqrf_mgpu       -c' + mn,
	'./testing_zgegqr_gpu        -c' + mn,
	
# ----------
# QR, CPU interface
	'./testing_zgelqf            -c' + mn,
	'./testing_zgeqlf            -c' + mn,
	'./testing_zgeqrf            -c' + mn,
	'./testing_zungqr            -c' + mnk,
	'./testing_zunmlq            -c' + mnk,
	'./testing_zunmql            -c' + mnk,
	'./testing_zunmqr            -c' + mnk,
	'./testing_zungqr_m          -c' + mnk,
	'./testing_zgeqp3            -c' + mn,
	'./testing_zgeqp3_gpu        -c' + mn,
	
# ----------
# symmetric eigenvalues, GPU interface
	'./testing_zheevd_gpu        -c' + n,  # does dsyevd_gpu
	'./testing_zhetrd_gpu        -c' + n,
	'./testing_zhetrd_mgpu       -c' + n,
	
# ----------
# symmetric eigenvalues, CPU interface
	'./testing_zheevd            -c' + n,  # does dsyevd
	'./testing_zhetrd            -c' + n,
	
# ----------
# symmetric eigenvalues, 2-stage
	'./testing_zhetrd_he2hb      -c' + n,
	'./testing_zheevdx_2stage    -c' + n,
	'./testing_zhetrd_he2hb_mgpu -c' + n,
	
# ----------
# generalized symmetric eigenvalues
	'./testing_zhegvd            -c' + n,  # does dsygvd
	'./testing_zhegvd_m          -c' + n,
	'./testing_zhegvdx           -c' + n,
	'./testing_zhegvdx_2stage    -c' + n,
	'./testing_zhegvdx_2stage_m  -c' + n,
	
# ----------
# non-symmetric eigenvalues
	'./testing_zgeev     -RV -LV -c' + n,  # does dgeev
	'./testing_zgehrd            -c' + n,
	'./testing_zgeev_m   -RV -LV -c' + n,  # does dgeev_m
	'./testing_zgehrd_m          -c' + n,
	
# ----------
# SVD
	'./testing_zgesdd    -US -VS -c' + mn,
	'./testing_zgesvd    -US -VS -c' + mn,
	'./testing_zgebrd            -c' + mn,
	'./testing_zunmbr            -c' + mnk,
);


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
nfail = 0
failures = []

start = None
if ( opts.start ):
	start = re.compile( opts.start + r'\b' )

seen = {}

for t in tests:
	
	make = False
	for p in opts.precisions:
		t2 = substitute( t, 'z', p )
		
		# skip tests before start
		if ( start and not start.search( t  )
		           and not start.search( t2 )):
			continue
		start = None
		
		# skip tests not in args, or duplicates, or non-existing
		m = re.search( '^./(\w*)', t2 )
		cmd = m.group(1)
		if ( args and not cmd in args ):
			continue
		if ( seen.has_key( cmd ) or not os.path.exists( cmd )):
			continue
		seen[cmd] = True
		
		# compress spaces
		t2 = re.sub( '  +', ' ', t2 )
		
		go = True
		while( go ):
			print
			print '*'*100
			print t2
			print '*'*100
			
			if ( make ):
				m = 'make ' + cmd
				print m
				os.system( m )
			# end
			
			err = os.system( t2 )
			if ( err > 0 ):
				n = (err >> 8)
				if ( n > 0 ):
					print n, 'tests failed'
					nfail += n
				else:
					print '\n>>> unknown error occured:', err, '<<<\n'
					nfail += 1
				# end
				failures.append( cmd )
			# end
			
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
