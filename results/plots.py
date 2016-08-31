# Script to plot MAGMA performance data. See help below and readme.txt.
# @author Mark Gates

from __future__ import print_function

import matplotlib.pyplot as pp
import matplotlib
import numpy
from numpy import isnan, zeros, where, linspace

import util


# ----------------------------------------------------------------------
# import versions
import v150_cuda70_k40c
import v160_cuda70_k40c
import v161_cuda70_k40c
import v162_cuda70_k40c  # same as v161 except sy/heevd
import v170_cuda70_k40c
import v200_cuda70_k40c

versions = [
	v150_cuda70_k40c,
	v160_cuda70_k40c,
	v161_cuda70_k40c,
	v162_cuda70_k40c,
	v170_cuda70_k40c,
	v200_cuda70_k40c,
]

# add local if it exists
try:
	import local
	versions.append( local )
	local.version = 'local'
except ImportError:
	pass

versions = numpy.array( versions )


# ----------------------------------------------------------------------
# setup colors for versions

# get nice distribution of colors from purple (old versions) to red (new versions)
#x = linspace( 0, 1, len(versions) )
#cool = matplotlib.cm.get_cmap('cool')
#colors = cool(x)

colors = [
 'r',
 'orange',
 'c',
 'b',
 'g',
 'm',
 'indigo',
 'SlateBlue',
 'LimeGreen',
]

# set colors, with latest being red (first)
for i in xrange( len(versions) ):
	#versions[i].color = colors[i,:]
	versions[i].color = colors[ len(versions) - 1 - i ]
# end


# ----------------------------------------------------------------------
# defaults
pp.rcParams['axes.axisbelow']   = True
pp.rcParams['figure.facecolor'] = 'white'
pp.rcParams['font.size']        = 10
pp.rcParams['legend.fontsize']  = 9

g_figsize  = [9, 7]
g_figsize2 = [6, 4]


# ----------------------------------------------------------------------
if ( not locals().has_key('g_subplots')):
	g_subplots = True
if ( not locals().has_key('g_figure')):
	g_figure   = None
if ( not locals().has_key('g_log')):
	g_log      = True
if ( not locals().has_key('g_save')):
	g_save     = False


# --------------------
def figure( fig ):
	global g_figure
	g_figure = fig
	if ( g_subplots ):
		pp.figure( fig )
# end


# --------------------
def clf( rows=0, cols=0 ):
	if ( g_subplots or rows == 0 or cols == 0 ):
		pp.clf()
	else:
		for index in range( 1, rows*cols+1 ):
			subplot( rows, cols, index )
			pp.clf()
# end


# --------------------
def subplot( rows, cols, index ):
	if ( g_subplots ):
		pp.subplot( rows, cols, index )
	else:
		pp.figure( g_figure*10 + index )
# end


# --------------------
def resize( size, rows=0, cols=0 ):
	if ( g_subplots or rows == 0 or cols == 0 ):
		util.resize( size )
	else:
		size2 = [ size[0]/cols, size[1]/rows ]
		for index in range( 1, rows*cols+1 ):
			subplot( rows, cols, index )
			util.resize( g_figsize2 )
# end


# --------------------
def savefig( name, rows=0, cols=0 ):
	if ( g_save ):
		if ( g_subplots or rows == 0 or cols == 0 ):
			print( 'saving', name )
			pp.savefig( name )
		else:
			prec = ['s', 'd', 'c', 'z']
			for index in range( 1, rows*cols+1 ):
				pname = prec[index-1] + name
				print( 'saving', pname )
				subplot( rows, cols, index )
				pp.savefig( pname )
# end


# --------------------
def set_title( versions, title ):
	v0 = versions[0]
	t = None
	for v in versions[1:]:
		if (v.cpu != v0.cpu) or (v.device != v0.device):
			t = 'various CPU & GPU'
			break
	if ( t == None ):
		t = 'GPU: ' + v0.device + ', CPU: ' + v0.cpu
	if ( title ):
		t = title + '\n' + t
	pp.title( t, fontsize=9 )
# end


# --------------------
def set_xticks():
	if ( g_log ):
		pp.xlabel( r'matrix dimension (log scale)' )
		xticks = [ 10, 100, 1000, 10000, 20000 ]
		xtickl = [ 10, 100, '1k', '10k', '20k' ]
	else:
		pp.xlabel( r'matrix dimension' )
		xticks = range( 0, 20001, 4000 )
		xtickl = xticks
	pp.xticks( xticks, xtickl )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end


# --------------------
# returns list of versions -- either original versions, or if it is single module, stick it in a list
module = type(numpy)  # why doesn't python know "module"?
def get_versions( versions ):
	if ( type(versions) == module ):
		return [versions]
	else:
		return versions
# end


# ----------------------------------------------------------------------
# column indices
getrf_m         = 0
getrf_n         = 1
getrf_cpu_flops = 2
getrf_cpu_time  = 3
getrf_gpu_flops = 4
getrf_gpu_time  = 5
getrf_error     = 6

potrf_n         = 0
potrf_cpu_flops = 1
potrf_cpu_time  = 2
potrf_gpu_flops = 3
potrf_gpu_time  = 4
potrf_error     = 5

geqrf_m         = 0
geqrf_n         = 1
geqrf_cpu_flops = 2
geqrf_cpu_time  = 3
geqrf_gpu_flops = 4
geqrf_gpu_time  = 5
geqrf_error     = 6

geev_n          = 0
geev_cpu_time   = 1
geev_gpu_time   = 2
geev_error      = 3

syev_n          = 0
syev_cpu_time   = 1
syev_gpu_time   = 2
syev_error      = 3

# note: testing_.gesvd outputs jobu & jobv; we throw out jobv in python data file
# to match testing_.gesdd which has only job.
svd_job         = 0
svd_m           = 1
svd_n           = 2
svd_cpu_time    = 3
svd_gpu_time    = 4
svd_error       = 5

symv_n             = 0
symv_gpu_flops     = 1
symv_gpu_time      = 2
symv_atomics_flops = 3
symv_atomics_time  = 4
symv_cublas_flops  = 5
symv_cublas_time   = 6
symv_cpu_flops     = 7
symv_cpu_time      = 8
symv_gpu_error     = 9
symv_atomics_error = 10
symv_cublas_error  = 11


# ----------------------------------------------------------------------
def str_none( s ):
	return str(s) if s is not None else ''


def plot_getrf_data( data, style='.-', color='y', label=None, idx=getrf_gpu_flops ):
	if ( g_log ):
		pp.semilogx( data[:,getrf_m], data[:,idx], style, color=color, lw=1.5, label=label )
	else:
		pp.plot(     data[:,getrf_m], data[:,idx], style, color=color, lw=1.5, label=label )
# end

def plot_getrf_labels( versions, title=None, legend=True ):
	set_title( versions, 'LU factorization ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	pp.ylabel( r'Gflop/s  $\frac{2}{3} n^3 / t$' )
	set_xticks()
# end

def plot_getrf( versions, cpu=True, gpu=True, lapack=True ):
	print( 'plotting getrf' )
	versions = get_versions( versions )
	figure( 1 )
	clf( 2, 2 )
	
	for v in versions:
		# ----- cpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgetrf') and cpu:
			plot_getrf_data( v.sgetrf,     '.-', color=v.color, label=v.version+' sgetrf'     )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgetrf') and cpu:
			plot_getrf_data( v.dgetrf,     '.-', color=v.color, label=v.version+' dgetrf'     )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgetrf') and cpu:
			plot_getrf_data( v.cgetrf,     '.-', color=v.color, label=v.version+' cgetrf'     )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgetrf') and cpu:
			plot_getrf_data( v.zgetrf,     '.-', color=v.color, label=v.version+' zgetrf'     )
		
		# ----- gpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgetrf_gpu') and gpu:
			plot_getrf_data( v.sgetrf_gpu, 'x-', color=v.color, label=v.version+' sgetrf_gpu' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgetrf_gpu') and gpu:
			plot_getrf_data( v.dgetrf_gpu, 'x-', color=v.color, label=v.version+' dgetrf_gpu' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgetrf_gpu') and gpu:
			plot_getrf_data( v.cgetrf_gpu, 'x-', color=v.color, label=v.version+' cgetrf_gpu' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgetrf_gpu') and gpu:
			plot_getrf_data( v.zgetrf_gpu, 'x-', color=v.color, label=v.version+' zgetrf_gpu' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('sgetrf') and not isnan( v.sgetrf[0,getrf_cpu_flops] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('sgetrf'):
				plot_getrf_data( v.sgetrf, 'k+-', color='k', label='MKL sgetrf', idx=getrf_cpu_flops )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dgetrf'):
				plot_getrf_data( v.dgetrf, 'k+-', color='k', label='MKL dgetrf', idx=getrf_cpu_flops )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cgetrf'):
				plot_getrf_data( v.cgetrf, 'k+-', color='k', label='MKL cgetrf', idx=getrf_cpu_flops )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zgetrf'):
				plot_getrf_data( v.zgetrf, 'k+-', color='k', label='MKL zgetrf', idx=getrf_cpu_flops )
			break
	# end
	
	subplot( 2, 2, 1 )
	plot_getrf_labels( versions, 'sgetrf', legend= True )
	
	subplot( 2, 2, 2 )
	plot_getrf_labels( versions, 'dgetrf', legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_getrf_labels( versions, 'cgetrf', legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_getrf_labels( versions, 'zgetrf', legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'getrf.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_potrf_data( data, style='.-', color='y', label=None, idx=potrf_gpu_flops ):
	if ( g_log ):
		pp.semilogx( data[:,potrf_n], data[:,idx], style, color=color, lw=1.5, label=label )
	else:
		pp.plot(     data[:,potrf_n], data[:,idx], style, color=color, lw=1.5, label=label )
# end

def plot_potrf_labels( versions, title=None, legend=True ):
	set_title( versions, 'Cholesky factorization ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	pp.ylabel( r'Gflop/s  $\frac{1}{3} n^3 / t$' )
	set_xticks()
# end

def plot_potrf( versions, cpu=True, gpu=True, lapack=True ):
	print( 'plotting potrf' )
	versions = get_versions( versions )
	figure( 2 )
	clf( 2, 2 )
	
	for v in versions:
		# ----- cpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('spotrf') and cpu:
			plot_potrf_data( v.spotrf,     '.-', color=v.color, label=v.version+' spotrf'     )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dpotrf') and cpu:
			plot_potrf_data( v.dpotrf,     '.-', color=v.color, label=v.version+' dpotrf'     )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cpotrf') and cpu:
			plot_potrf_data( v.cpotrf,     '.-', color=v.color, label=v.version+' cpotrf'     )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zpotrf') and cpu:
			plot_potrf_data( v.zpotrf,     '.-', color=v.color, label=v.version+' zpotrf'     )
		
		# ----- gpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('spotrf_gpu') and gpu:
			plot_potrf_data( v.spotrf_gpu, 'x-', color=v.color, label=v.version+' spotrf_gpu' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dpotrf_gpu') and gpu:
			plot_potrf_data( v.dpotrf_gpu, 'x-', color=v.color, label=v.version+' dpotrf_gpu' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cpotrf_gpu') and gpu:
			plot_potrf_data( v.cpotrf_gpu, 'x-', color=v.color, label=v.version+' cpotrf_gpu' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zpotrf_gpu') and gpu:
			plot_potrf_data( v.zpotrf_gpu, 'x-', color=v.color, label=v.version+' zpotrf_gpu' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('spotrf') and not isnan( v.spotrf[0,potrf_cpu_flops] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('spotrf'):
				plot_potrf_data( v.spotrf, 'k+-', color='k', label='MKL spotrf', idx=potrf_cpu_flops )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dpotrf'):
				plot_potrf_data( v.dpotrf, 'k+-', color='k', label='MKL dpotrf', idx=potrf_cpu_flops )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cpotrf'):
				plot_potrf_data( v.cpotrf, 'k+-', color='k', label='MKL cpotrf', idx=potrf_cpu_flops )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zpotrf'):
				plot_potrf_data( v.zpotrf, 'k+-', color='k', label='MKL zpotrf', idx=potrf_cpu_flops )
			break
	# end
	
	subplot( 2, 2, 1 )
	plot_potrf_labels( versions, 'spotrf', legend= True )
	
	subplot( 2, 2, 2 )
	plot_potrf_labels( versions, 'dpotrf', legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_potrf_labels( versions, 'cpotrf', legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_potrf_labels( versions, 'zpotrf', legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'potrf.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_geqrf_data( data, style='.-', color='y', label=None, idx=geqrf_gpu_flops ):
	if ( g_log ):
		pp.semilogx( data[:,geqrf_m], data[:,idx], style, color=color, lw=1.5, label=label )
	else:
		pp.plot(     data[:,geqrf_m], data[:,idx], style, color=color, lw=1.5, label=label )
# end

def plot_geqrf_labels( versions, title=None, legend=True ):
	set_title( versions, 'QR factorization ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	pp.ylabel( r'Gflop/s  $\frac{4}{3} n^3 / t$' )
	set_xticks()
# end

def plot_geqrf( versions, cpu=True, gpu=True, lapack=True ):
	print( 'plotting geqrf' )
	versions = get_versions( versions )
	figure( 3 )
	clf( 2, 2 )
	
	for v in versions:
		# ----- cpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgeqrf') and cpu:
			plot_geqrf_data( v.sgeqrf,     '.-', color=v.color, label=v.version+' sgeqrf'     )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgeqrf') and cpu:
			plot_geqrf_data( v.dgeqrf,     '.-', color=v.color, label=v.version+' dgeqrf'     )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgeqrf') and cpu:
			plot_geqrf_data( v.cgeqrf,     '.-', color=v.color, label=v.version+' cgeqrf'     )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgeqrf') and cpu:
			plot_geqrf_data( v.zgeqrf,     '.-', color=v.color, label=v.version+' zgeqrf'     )
		
		# ----- gpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgeqrf_gpu') and gpu:
			plot_geqrf_data( v.sgeqrf_gpu, 'x-', color=v.color, label=v.version+' sgeqrf_gpu' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgeqrf_gpu') and gpu:
			plot_geqrf_data( v.dgeqrf_gpu, 'x-', color=v.color, label=v.version+' dgeqrf_gpu' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgeqrf_gpu') and gpu:
			plot_geqrf_data( v.cgeqrf_gpu, 'x-', color=v.color, label=v.version+' cgeqrf_gpu' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgeqrf_gpu') and gpu:
			plot_geqrf_data( v.zgeqrf_gpu, 'x-', color=v.color, label=v.version+' zgeqrf_gpu' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('sgeqrf') and not isnan( v.sgeqrf[0,geqrf_cpu_flops] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('sgeqrf'):
				plot_geqrf_data( v.sgeqrf, 'k+-', color='k', label='MKL sgeqrf', idx=geqrf_cpu_flops )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dgeqrf'):
				plot_geqrf_data( v.dgeqrf, 'k+-', color='k', label='MKL dgeqrf', idx=geqrf_cpu_flops )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cgeqrf'):
				plot_geqrf_data( v.cgeqrf, 'k+-', color='k', label='MKL cgeqrf', idx=geqrf_cpu_flops )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zgeqrf'):
				plot_geqrf_data( v.zgeqrf, 'k+-', color='k', label='MKL zgeqrf', idx=geqrf_cpu_flops )
			break
	# end
	
	subplot( 2, 2, 1 )
	plot_geqrf_labels( versions, 'sgeqrf', legend= True )
	
	subplot( 2, 2, 2 )
	plot_geqrf_labels( versions, 'dgeqrf', legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_geqrf_labels( versions, 'cgeqrf', legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_geqrf_labels( versions, 'zgeqrf', legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'geqrf.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_geev_data( data, vec, style='.-', color='y', label=None, idx=geev_gpu_time ):
	n = data[:,geev_n]
	t = data[:,idx]
	if ( vec ):
		# 10/3 Hess + 10/3 QR iter + 4/3 generate Q + 4/3 right vectors
		gflop = 1e-9 * 28/3. * n**3
	else:
		# 10/3 Hess +  4/3 QR iter
		gflop = 1e-9 * 14/3. * n**3
	if ( g_log ):
		pp.semilogx( n, gflop/t, style, color=color, lw=1.5, label=label )
	else:
		pp.plot(     n, gflop/t, style, color=color, lw=1.5, label=label )
# end

def plot_geev_labels( versions, title, vec, legend=True ):
	set_title( versions, 'Non-symmetric eigenvalues, ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	if ( vec ):
		pp.ylabel( r'Gflop/s   $\frac{28}{3} n^3 / t$' )
	else:
		pp.ylabel( r'Gflop/s   $\frac{14}{3} n^3 / t$' )
	#pp.ylabel( 'time (sec)' )
	set_xticks()
# end

def plot_geev( versions, lapack=True ):
	print( 'plotting geev' )
	versions = get_versions( versions )
	figure( 4 )
	clf( 2, 2 )
	
	for v in versions:
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgeev_RN'):
			plot_geev_data(  v.sgeev_RN, False, '.-', color=v.color, label=v.version+' sgeev' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgeev_RN'):
			plot_geev_data(  v.dgeev_RN, False, '.-', color=v.color, label=v.version+' dgeev' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgeev_RN'):
			plot_geev_data(  v.cgeev_RN, False, '.-', color=v.color, label=v.version+' cgeev' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgeev_RN'):
			plot_geev_data(  v.zgeev_RN, False, '.-', color=v.color, label=v.version+' zgeev' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('sgeev_RN') and not isnan( v.sgeev_RN[0,geev_cpu_time] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('sgeev_RN'):
				plot_geev_data(  v.sgeev_RN, False, 'k+-', color='k', label='MKL sgeev', idx=geev_cpu_time )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dgeev_RN'):
				plot_geev_data(  v.dgeev_RN, False, 'k+-', color='k', label='MKL dgeev', idx=geev_cpu_time )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cgeev_RN'):
				plot_geev_data(  v.cgeev_RN, False, 'k+-', color='k', label='MKL cgeev', idx=geev_cpu_time )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zgeev_RN'):
				plot_geev_data(  v.zgeev_RN, False, 'k+-', color='k', label='MKL zgeev', idx=geev_cpu_time )
			break
	# end
	
	subplot( 2, 2, 1 )
	plot_geev_labels( versions, 'sgeev, no vectors', vec=False, legend= True )
	
	subplot( 2, 2, 2 )
	plot_geev_labels( versions, 'dgeev, no vectors', vec=False, legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_geev_labels( versions, 'cgeev, no vectors', vec=False, legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_geev_labels( versions, 'zgeev, no vectors', vec=False, legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'geev_rn.pdf', 2, 2 )
	
	# --------------------
	figure( 5 )
	clf( 2, 2 )
	
	for v in versions:
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgeev_RV'):
			plot_geev_data(  v.sgeev_RV, True, '.-', color=v.color, label=v.version+' sgeev' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgeev_RV'):
			plot_geev_data(  v.dgeev_RV, True, '.-', color=v.color, label=v.version+' dgeev' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgeev_RV'):
			plot_geev_data(  v.cgeev_RV, True, '.-', color=v.color, label=v.version+' cgeev' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgeev_RV'):
			plot_geev_data(  v.zgeev_RV, True, '.-', color=v.color, label=v.version+' zgeev' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('sgeev_RV') and not isnan( v.sgeev_RV[0,geev_cpu_time] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('sgeev_RV'):
				plot_geev_data(  v.sgeev_RV, True, 'k+-', color='k', label='MKL sgeev', idx=geev_cpu_time )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dgeev_RV'):
				plot_geev_data(  v.dgeev_RV, True, 'k+-', color='k', label='MKL dgeev', idx=geev_cpu_time )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cgeev_RV'):
				plot_geev_data(  v.cgeev_RV, True, 'k+-', color='k', label='MKL cgeev', idx=geev_cpu_time )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zgeev_RV'):
				plot_geev_data(  v.zgeev_RV, True, 'k+-', color='k', label='MKL zgeev', idx=geev_cpu_time )
			break
	# end
	
	subplot( 2, 2, 1 )
	plot_geev_labels( versions, 'sgeev, with right vectors', vec=True, legend= True )
	
	subplot( 2, 2, 2 )
	plot_geev_labels( versions, 'dgeev, with right vectors', vec=True, legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_geev_labels( versions, 'cgeev, with right vectors', vec=True, legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_geev_labels( versions, 'zgeev, with right vectors', vec=True, legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'geev_rv.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_syev_data( data, vec, style='.-', color='y', label=None, idx=syev_gpu_time ):
	n = data[:,syev_n]
	t = data[:,idx]  # idx is syev_cpu_time or syev_gpu_time
	if ( vec ):
		# 4/3 sytrd + 4/3 solve + 2 mqr
		gflop = 1e-9 * 14/3. * n**3
	else:
		gflop = 1e-9 * 4/3. * n**3
	
	if ( g_log ):
		pp.semilogx( n, gflop/t, style, color=color, lw=1.5, label=label )
	else:
		pp.plot(     n, gflop/t, style, color=color, lw=1.5, label=label )
# end

def plot_syev_labels( versions, title, vec, legend=True ):
	set_title( versions, 'Symmetric eigenvalues, ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	if ( vec ):
		pp.ylabel( r'Gflop/s   $\frac{14}{3} n^3 / t$' )
	else:
		pp.ylabel( r'Gflop/s   $\frac{4}{3} n^3 / t$' )
	#pp.ylabel( 'time (sec)' )
	set_xticks()
# end

def plot_syev( versions, cpu=True, gpu=True, bulge=True, lapack=True ):
	print( 'plotting syev' )
	versions = get_versions( versions )
	figure( 6 )
	clf( 2, 2 )
	
	for v in versions:
		# ---- bulge
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssyevdx_2stage_JN') and bulge:
			plot_syev_data(  v.ssyevdx_2stage_JN, False, 's-', color=v.color, label=v.version+' ssyevdx_2stage' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsyevdx_2stage_JN') and bulge:
			plot_syev_data(  v.dsyevdx_2stage_JN, False, 's-', color=v.color, label=v.version+' dsyevdx_2stage' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cheevdx_2stage_JN') and bulge:
			plot_syev_data(  v.cheevdx_2stage_JN, False, 's-', color=v.color, label=v.version+' cheevdx_2stage' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zheevdx_2stage_JN') and bulge:
			plot_syev_data(  v.zheevdx_2stage_JN, False, 's-', color=v.color, label=v.version+' zheevdx_2stage' )
		
		# ----- cpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssyevd_JN') and cpu:
			plot_syev_data(  v.ssyevd_JN,     False, '.-', color=v.color, label=v.version+' ssyevd'     )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsyevd_JN') and cpu:
			plot_syev_data(  v.dsyevd_JN,     False, '.-', color=v.color, label=v.version+' dsyevd'     )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cheevd_JN') and cpu:
			plot_syev_data(  v.cheevd_JN,     False, '.-', color=v.color, label=v.version+' cheevd'     )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zheevd_JN') and cpu:
			plot_syev_data(  v.zheevd_JN,     False, '.-', color=v.color, label=v.version+' zheevd'     )
		
		# ----- gpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssyevd_gpu_JN') and gpu:
			plot_syev_data(  v.ssyevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' ssyevd_gpu' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsyevd_gpu_JN') and gpu:
			plot_syev_data(  v.dsyevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' dsyevd_gpu' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cheevd_gpu_JN') and gpu:
			plot_syev_data(  v.cheevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' cheevd_gpu' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zheevd_gpu_JN') and gpu:
			plot_syev_data(  v.zheevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' zheevd_gpu' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('ssyevd_JN') and not isnan( v.ssyevd_JN[0,syev_cpu_time] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('ssyevd_JN'):
				plot_syev_data(  v.ssyevd_JN,     False, 'k+-', color='k', label='MKL ssyevd', idx=syev_cpu_time )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dsyevd_JN'):
				plot_syev_data(  v.dsyevd_JN,     False, 'k+-', color='k', label='MKL dsyevd', idx=syev_cpu_time )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cheevd_JN'):
				plot_syev_data(  v.cheevd_JN,     False, 'k+-', color='k', label='MKL cheevd', idx=syev_cpu_time )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zheevd_JN'):
				plot_syev_data(  v.zheevd_JN,     False, 'k+-', color='k', label='MKL zheevd', idx=syev_cpu_time )
			break
	# end
	
	subplot( 2, 2, 1 )
	plot_syev_labels( versions, 'ssyevd, no vectors', vec=False, legend= True )
	
	subplot( 2, 2, 2 )
	plot_syev_labels( versions, 'dsyevd, no vectors', vec=False, legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_syev_labels( versions, 'cheevd, no vectors', vec=False, legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_syev_labels( versions, 'zheevd, no vectors', vec=False, legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'syev_jn.pdf', 2, 2 )
	
	# --------------------
	figure( 7 )
	clf( 2, 2 )
	
	for v in versions:
		# ----- bulge
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssyevdx_2stage_JV') and bulge:
			plot_syev_data(  v.ssyevdx_2stage_JV, True, 's-', color=v.color, label=v.version+' ssyevdx_2stage' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsyevdx_2stage_JV') and bulge:
			plot_syev_data(  v.dsyevdx_2stage_JV, True, 's-', color=v.color, label=v.version+' dsyevdx_2stage' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cheevdx_2stage_JV') and bulge:
			plot_syev_data(  v.cheevdx_2stage_JV, True, 's-', color=v.color, label=v.version+' cheevdx_2stage' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zheevdx_2stage_JV') and bulge:
			plot_syev_data(  v.zheevdx_2stage_JV, True, 's-', color=v.color, label=v.version+' zheevdx_2stage' )
		
		# ----- cpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssyevd_JV') and cpu:
			plot_syev_data(  v.ssyevd_JV,     True, '.-', color=v.color, label=v.version+' ssyevd'     )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsyevd_JV') and cpu:
			plot_syev_data(  v.dsyevd_JV,     True, '.-', color=v.color, label=v.version+' dsyevd'     )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cheevd_JV') and cpu:
			plot_syev_data(  v.cheevd_JV,     True, '.-', color=v.color, label=v.version+' cheevd'     )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zheevd_JV') and cpu:
			plot_syev_data(  v.zheevd_JV,     True, '.-', color=v.color, label=v.version+' zheevd'     )
		
		# ----- gpu
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssyevd_gpu_JV') and gpu:
			plot_syev_data(  v.ssyevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' ssyevd_gpu' )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsyevd_gpu_JV') and gpu:
			plot_syev_data(  v.dsyevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' dsyevd_gpu' )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cheevd_gpu_JV') and gpu:
			plot_syev_data(  v.cheevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' cheevd_gpu' )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zheevd_gpu_JV') and gpu:
			plot_syev_data(  v.zheevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' zheevd_gpu' )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('ssyevd_JV') and not isnan( v.ssyevd_JV[0,syev_cpu_time] )):
			print( 'found LAPACK in', v.version )
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('ssyevd_JV'):
				plot_syev_data(  v.ssyevd_JV,     True, 'k+-', color='k', label='MKL ssyevd', idx=syev_cpu_time )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dsyevd_JV'):
				plot_syev_data(  v.dsyevd_JV,     True, 'k+-', color='k', label='MKL dsyevd', idx=syev_cpu_time )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cheevd_JV'):
				plot_syev_data(  v.cheevd_JV,     True, 'k+-', color='k', label='MKL cheevd', idx=syev_cpu_time )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zheevd_JV'):
				plot_syev_data(  v.zheevd_JV,     True, 'k+-', color='k', label='MKL zheevd', idx=syev_cpu_time )
			break
	# end
	
	for i in xrange( 1, 5 ):
		subplot( 2, 2, 1 )
		plot_syev_labels( versions, 'ssyevd, with vectors', vec=True, legend= True )
		
		subplot( 2, 2, 2 )
		plot_syev_labels( versions, 'dsyevd, with vectors', vec=True, legend= not g_subplots )
		
		subplot( 2, 2, 3 )
		plot_syev_labels( versions, 'cheevd, with vectors', vec=True, legend= not g_subplots )
		
		subplot( 2, 2, 4 )
		plot_syev_labels( versions, 'zheevd, with vectors', vec=True, legend= not g_subplots )
	# end
	resize( g_figsize, 2, 2 )
	savefig( 'syev_jv.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_gesvd_data( data, vec, style='.-', color='y', label=None, ratio=1, idx=svd_gpu_time ):
	(ii,) = where( data[:,svd_m] == ratio*data[:,svd_n] )
	mn = zeros(( len(ii), 2 ))
	mn[:,0] = data[ii,svd_m]
	mn[:,1] = data[ii,svd_n]
	M = mn.max( axis=1 )  # M = max(m,n)
	N = mn.min( axis=1 )  # N = min(m,n)
	
	if ( vec ):
		# for D&C some vectors (no initial QR)
		gflop = 1e-9 * (8*M*N**2 + (4/3.)*N**3)
	else:
		gflop = 1e-9 * (4*M*N**2 - (4/3.)*N**3)
	t = data[ii,idx]
	
	if ( g_log ):
		pp.semilogx( N, gflop/t, style, color=color, lw=1.5, label=label )
	else:
		pp.plot(     N, gflop/t, style, color=color, lw=1.5, label=label )
# end

def plot_gesvd_labels( versions, title, vec, square, legend=True ):
	set_title( versions, 'SVD, ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	if ( vec ):
		if ( square ):
			pp.ylabel( r'Gflop/s   $9n^3/t$' )
		else:
			pp.ylabel( r'Gflop/s   $(8mn^2 + \frac{4}{3}n^3)/t$' )
	else:
		if ( square ):
			pp.ylabel( r'Gflop/s   $\frac{8}{3} n^3 / t$' )
		else:
			pp.ylabel( r'Gflop/s   $(4mn^2 - \frac{4}{3} n^3) / t$' )
	#pp.ylabel( 'time (sec)' )
	set_xticks()
# end

def plot_gesvd( versions, ratio=1, lapack=True, svd=True, sdd=True ):
	print( 'plotting gesvd' )
	versions = get_versions( versions )
	figure( 8 )
	clf( 2, 2 )
	
	for v in versions:
		# for no vectors, gesvd == gesdd
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgesdd_UN') and sdd:
			plot_gesvd_data( v.sgesdd_UN, False, 'x--', color=v.color, label=v.version+' sgesdd', ratio=ratio )
		if v.__dict__.has_key('sgesvd_UN') and svd:
			plot_gesvd_data( v.sgesvd_UN, False, '.-',  color=v.color, label=v.version+' sgesvd', ratio=ratio )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgesdd_UN') and sdd:
			plot_gesvd_data( v.dgesdd_UN, False, 'x--', color=v.color, label=v.version+' dgesdd', ratio=ratio )
		if v.__dict__.has_key('dgesvd_UN') and svd:
			plot_gesvd_data( v.dgesvd_UN, False, '.-',  color=v.color, label=v.version+' dgesvd', ratio=ratio )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgesdd_UN') and sdd:
			plot_gesvd_data( v.cgesdd_UN, False, 'x--', color=v.color, label=v.version+' cgesdd', ratio=ratio )
		if v.__dict__.has_key('cgesvd_UN') and svd:
			plot_gesvd_data( v.cgesvd_UN, False, '.-',  color=v.color, label=v.version+' cgesvd', ratio=ratio )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgesdd_UN') and sdd:
			plot_gesvd_data( v.zgesdd_UN, False, 'x--', color=v.color, label=v.version+' zgesdd', ratio=ratio )
		if v.__dict__.has_key('zgesvd_UN') and svd:
			plot_gesvd_data( v.zgesvd_UN, False, '.-',  color=v.color, label=v.version+' zgesvd', ratio=ratio )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('sgesdd_UN') and not isnan( v.sgesdd_UN[0,svd_cpu_time] )):
			print( 'found LAPACK in', v.version )
			
			# for no vectors, gesvd == gesdd
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('sgesdd_UN') and sdd:
				plot_gesvd_data( v.sgesdd_UN, False, 'x--', color='k', label='MKL sgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('sgesvd_UN') and svd:
				plot_gesvd_data( v.sgesvd_UN, False, '.-',  color='k', label='MKL sgesvd', ratio=ratio, idx=svd_cpu_time )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dgesdd_UN') and sdd:
				plot_gesvd_data( v.dgesdd_UN, False, 'x--', color='k', label='MKL dgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('dgesvd_UN') and svd:
				plot_gesvd_data( v.dgesvd_UN, False, '.-',  color='k', label='MKL dgesvd', ratio=ratio, idx=svd_cpu_time )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cgesdd_UN') and sdd:
				plot_gesvd_data( v.cgesdd_UN, False, 'x--', color='k', label='MKL cgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('cgesvd_UN') and svd:
				plot_gesvd_data( v.cgesvd_UN, False, '.-',  color='k', label='MKL cgesvd', ratio=ratio, idx=svd_cpu_time )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zgesdd_UN') and sdd:
				plot_gesvd_data( v.zgesdd_UN, False, 'x--', color='k', label='MKL zgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('zgesvd_UN') and svd:
				plot_gesvd_data( v.zgesvd_UN, False, '.-',  color='k', label='MKL zgesvd', ratio=ratio, idx=svd_cpu_time )
			break
	# end
	
	m = ratio if (ratio >= 1) else 1
	n = 1     if (ratio >= 1) else 1/ratio
	subplot( 2, 2, 1 )
	plot_gesvd_labels( versions, 'sgesvd, no vectors, M:N ratio %.3g:%.3g' % (m,n), vec=False, square=(ratio == 1), legend= True )
	
	subplot( 2, 2, 2 )
	plot_gesvd_labels( versions, 'dgesvd, no vectors, M:N ratio %.3g:%.3g' % (m,n), vec=False, square=(ratio == 1), legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_gesvd_labels( versions, 'cgesvd, no vectors, M:N ratio %.3g:%.3g' % (m,n), vec=False, square=(ratio == 1), legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_gesvd_labels( versions, 'zgesvd, no vectors, M:N ratio %.3g:%.3g' % (m,n), vec=False, square=(ratio == 1), legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'gesvd_jn.pdf', 2, 2 )
	
	# --------------------
	figure( 9 )
	clf( 2, 2 )
	
	for v in versions:
		# for vectors, gesdd > gesvd performance
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('sgesdd_US') and sdd:
			plot_gesvd_data( v.sgesdd_US, True, 'x--', color=v.color, label=v.version+' sgesdd', ratio=ratio )
		if v.__dict__.has_key('sgesvd_US') and svd:
			plot_gesvd_data( v.sgesvd_US, True, '.-',  color=v.color, label=v.version+' sgesvd', ratio=ratio )
			
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dgesdd_US') and sdd:
			plot_gesvd_data( v.dgesdd_US, True, 'x--', color=v.color, label=v.version+' dgesdd', ratio=ratio )
		if v.__dict__.has_key('dgesvd_US') and svd:
			plot_gesvd_data( v.dgesvd_US, True, '.-',  color=v.color, label=v.version+' dgesvd', ratio=ratio )
			
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('cgesdd_US') and sdd:
			plot_gesvd_data( v.cgesdd_US, True, 'x--', color=v.color, label=v.version+' cgesdd', ratio=ratio )
		if v.__dict__.has_key('cgesvd_US') and svd:
			plot_gesvd_data( v.cgesvd_US, True, '.-',  color=v.color, label=v.version+' cgesvd', ratio=ratio )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zgesdd_US') and sdd:
			plot_gesvd_data( v.zgesdd_US, True, 'x--', color=v.color, label=v.version+' zgesdd', ratio=ratio )
		if v.__dict__.has_key('zgesvd_US') and svd:
			plot_gesvd_data( v.zgesvd_US, True, '.-',  color=v.color, label=v.version+' zgesvd', ratio=ratio )
	# end
	
	# plot lapack last; stop after 1st occurence
	for v in versions[-1::-1]:
		if ( lapack and v.__dict__.has_key('sgesdd_US') and not isnan( v.sgesdd_US[0,svd_cpu_time] )):
			print( 'found LAPACK in', v.version )
			
			# for vectors, gesdd > gesvd performance
			subplot( 2, 2, 1 )
			if v.__dict__.has_key('sgesdd_US') and sdd:
				plot_gesvd_data( v.sgesdd_US, True, 'x--', color='k', label='MKL sgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('sgesvd_US') and svd:
				plot_gesvd_data( v.sgesvd_US, True, '.-',  color='k', label='MKL sgesvd', ratio=ratio, idx=svd_cpu_time )
			
			subplot( 2, 2, 2 )
			if v.__dict__.has_key('dgesdd_US') and sdd:
				plot_gesvd_data( v.dgesdd_US, True, 'x--', color='k', label='MKL dgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('dgesvd_US') and svd:
				plot_gesvd_data( v.dgesvd_US, True, '.-',  color='k', label='MKL dgesvd', ratio=ratio, idx=svd_cpu_time )
			
			subplot( 2, 2, 3 )
			if v.__dict__.has_key('cgesdd_US') and sdd:
				plot_gesvd_data( v.cgesdd_US, True, 'x--', color='k', label='MKL cgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('cgesvd_US') and svd:
				plot_gesvd_data( v.cgesvd_US, True, '.-',  color='k', label='MKL cgesvd', ratio=ratio, idx=svd_cpu_time )
			
			subplot( 2, 2, 4 )
			if v.__dict__.has_key('zgesdd_US') and sdd:
				plot_gesvd_data( v.zgesdd_US, True, 'x--', color='k', label='MKL zgesdd', ratio=ratio, idx=svd_cpu_time )
			if v.__dict__.has_key('zgesvd_US') and svd:
				plot_gesvd_data( v.zgesvd_US, True, '.-',  color='k', label='MKL zgesvd', ratio=ratio, idx=svd_cpu_time )
			break
	# end
	
	m = ratio if (ratio >= 1) else 1
	n = 1     if (ratio >= 1) else 1/ratio
	subplot( 2, 2, 1 )
	plot_gesvd_labels( versions, 'sgesvd, some vectors, M:N ratio %.3g:%.3g' % (m,n), vec=True, square=(ratio == 1), legend= True )
	
	subplot( 2, 2, 2 )
	plot_gesvd_labels( versions, 'dgesvd, some vectors, M:N ratio %.3g:%.3g' % (m,n), vec=True, square=(ratio == 1), legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_gesvd_labels( versions, 'cgesvd, some vectors, M:N ratio %.3g:%.3g' % (m,n), vec=True, square=(ratio == 1), legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_gesvd_labels( versions, 'zgesvd, some vectors, M:N ratio %.3g:%.3g' % (m,n), vec=True, square=(ratio == 1), legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'gesvd_js.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_symv_data( data, uplo, style='.-', color='y', label=None, first=False ):
	uplo += ' '
	if ( first ):
		if ( g_log ):
			pp.semilogx( data[:,symv_n], data[:,symv_atomics_flops], style, markevery=10, color='#aaaaaa', lw=1.5, label=uplo + 'cublas atomics' )
			pp.semilogx( data[:,symv_n], data[:,symv_cublas_flops],  style, markevery=10, color='#666666', lw=1.5, label=uplo + 'cublas'         )
			pp.semilogx( data[:,symv_n], data[:,symv_cpu_flops],     style, markevery=10, color='black',   lw=1.5, label=uplo + 'MKL'            )
		else:
			pp.plot(     data[:,symv_n], data[:,symv_atomics_flops], style, markevery=10, color='#aaaaaa', lw=1.5, label=uplo + 'cublas atomics' )
			pp.plot(     data[:,symv_n], data[:,symv_cublas_flops],  style, markevery=10, color='#666666', lw=1.5, label=uplo + 'cublas'         )
			pp.plot(     data[:,symv_n], data[:,symv_cpu_flops],     style, markevery=10, color='black',   lw=1.5, label=uplo + 'MKL'            )
	if ( g_log ):
		pp.semilogx( data[:,symv_n], data[:,symv_gpu_flops], style, markevery=10, color=color, lw=1.5, label=uplo + label )
	else:
		pp.plot(     data[:,symv_n], data[:,symv_gpu_flops], style, markevery=10, color=color, lw=1.5, label=uplo + label )
# end

def plot_symv_labels( versions, title=None, legend=True ):
	set_title( versions, 'Symmetric matrix-vector multiply (symv) ' + str_none(title) )
	if legend:
		pp.legend( loc='upper left' )
	pp.ylabel( r'Gflop/s  $2n^2 / t$' )
	set_xticks()
	#(ymin, ymax) = pp.ylim()
	#pp.ylim( 0, min(310, ymax) )
# end

def plot_symv( versions ):
	print( 'plotting symv' )
	versions = get_versions( versions )
	figure( 10 )
	clf( 2, 2 )
	
	first = True
	for v in versions:
		subplot( 2, 2, 1 )
		if v.__dict__.has_key('ssymv_L'):
			plot_symv_data(  v.ssymv_L, 'lower', '-',   color=v.color, label=v.version+' ssymv', first=first )
		if v.__dict__.has_key('ssymv_U'):
			plot_symv_data(  v.ssymv_U, 'upper', 'x--', color=v.color, label=v.version+' ssymv', first=first )
		
		subplot( 2, 2, 2 )
		if v.__dict__.has_key('dsymv_L'):
			plot_symv_data(  v.dsymv_L, 'lower', '-',   color=v.color, label=v.version+' dsymv', first=first )
		if v.__dict__.has_key('dsymv_U'):
			plot_symv_data(  v.dsymv_U, 'upper', 'x--', color=v.color, label=v.version+' dsymv', first=first )
		
		subplot( 2, 2, 3 )
		if v.__dict__.has_key('chemv_L'):
			plot_symv_data(  v.chemv_L, 'lower', '-',   color=v.color, label=v.version+' chemv', first=first )
		if v.__dict__.has_key('chemv_U'):
			plot_symv_data(  v.chemv_U, 'upper', 'x--', color=v.color, label=v.version+' chemv', first=first )
		
		subplot( 2, 2, 4 )
		if v.__dict__.has_key('zhemv_L'):
			plot_symv_data(  v.zhemv_L, 'lower', '-',   color=v.color, label=v.version+' zhemv', first=first )
		if v.__dict__.has_key('zhemv_U'):
			plot_symv_data(  v.zhemv_U, 'upper', 'x--', color=v.color, label=v.version+' zhemv', first=first )
		first = False
	# end
	
	subplot( 2, 2, 1 )
	plot_symv_labels( versions, 'ssymv', legend= True )
	
	subplot( 2, 2, 2 )
	plot_symv_labels( versions, 'dsymv', legend= not g_subplots )
	
	subplot( 2, 2, 3 )
	plot_symv_labels( versions, 'chemv', legend= not g_subplots )
	
	subplot( 2, 2, 4 )
	plot_symv_labels( versions, 'zhemv', legend= not g_subplots )
	
	resize( g_figsize, 2, 2 )
	savefig( 'symv.pdf', 2, 2 )
# end


# ----------------------------------------------------------------------
def plot_all( versions, lapack=True, cpu=True, gpu=True, bulge=True, sdd=True, svd=True, ratio=1 ):
	plot_getrf( versions, lapack=lapack, cpu=cpu, gpu=gpu )
	plot_potrf( versions, lapack=lapack, cpu=cpu, gpu=gpu )
	plot_geqrf( versions, lapack=lapack, cpu=cpu, gpu=gpu )
	plot_geev(  versions, lapack=lapack )
	plot_syev(  versions, lapack=lapack, cpu=cpu, gpu=gpu, bulge=bulge )
	plot_gesvd( versions, lapack=lapack, ratio=ratio, svd=svd, sdd=sdd )
	plot_symv(  versions )
# end


# ----------------------------------------------------------------------
def print_help():
	print('''Global settings:
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

Available versions:''')
	
	for i in xrange( len(versions) ):
		print( "versions[%d] = %s" % (i, versions[i].version) )
	# end
# end


if ( __name__ == '__main__' ):
	print_help()
