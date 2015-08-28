import re
import numpy


# ------------------------------------------------------------
def intersect( A, B ):
	'''
	Finds common elements in A and B
	Returns pair of index arrays (ii, jj) such that a[ii] == b[jj].
	'''
	(ii,) = numpy.where( numpy.in1d( A, B ))
	(jj,) = numpy.where( numpy.in1d( B, A ))
	return (ii, jj)
# end


# ------------------------------------------------------------
def groupby( data, index ):
	'''
	Returns array with unique values in column index,
	and min-reduction in all other columns, like an SQL groupby query.
	'''
	if ( data is None ):
		return None
	
	values = numpy.unique( data[:,index] )
	rows = len(values)
	cols = data.shape[1]
	data2 = numpy.zeros(( rows, cols ))
	for i in xrange( rows ):
		(ii,) = numpy.where( data[:,index] == values[i] )
		for j in xrange( cols ):
			data2[i,j] = numpy.min( data[ii,j] )
	# end
	return data2
# end


# ------------------------------------------------------------
def legend( lines=None, labels=None, loc="best", **kwargs ):
	'''
	Wrapper around pyplot's legend
	Adds "[upper,lower] outside" and "outside [left,right]" as locations.
	
	Adds left, right, upper, lower as aliases to
	to   center left, center right, upper center, lower center, respectively.
	
	Accepts locations with word order transposed,
	e.g., "center lower" instead of "lower center".
	'''
	import matplotlib.pyplot as pp
	
	anchor = None
	locs = re.split( ' +', loc )
	if ( len(locs) == 1 ):
		if   ( locs[0] in ('upper', 'top'   )): locs = ['upper center']
		elif ( locs[0] in ('lower', 'bottom')): locs = ['lower center']
		elif ( locs[0] in ('left'           )): locs = ['center left']
		elif ( locs[0] not in ('right', 'best', 'center')):
			print 'loc=\'' + loc + '\' is invalid; should be one of:'
			print 'best,'
			print '                           upper  outside'
			print '              upper  left, upper  center (upper), upper  right,'
			print 'outside left, center left,        center,         center right, outside right'
			print '                (left)                              (right)'
			print '              lower  left, lower  center (lower), lower  right,'
			print '                           lower  outside'
			print '(top=upper and bottom=lower are recognized synonyms)'
			locs = ['best']
	elif ( len(locs) == 2 ):
		if ( locs[0] in ('left', 'right') or locs[1] in ('lower', 'upper', 'top', 'bottom')):
			locs = [locs[1], locs[0]]  # swap
		if ( locs[0] == 'top'    ): locs[0] = 'upper'
		if ( locs[0] == 'bottom' ): locs[0] = 'lower'
	else:
		raise Exception()
	
	loc2 = ' '.join( locs )
	if ( loc != loc2 ):
		print 'using loc=\'' + loc2 + '\''
	
	if ( loc2 == 'outside right' ):
		anchor = (1, 0.5)
		loc2 = 'center left'
	elif ( loc2 == 'outside left' ):
		anchor = (-0.05, 0.5)  # depends on figure size, which is horrible
		loc2 = 'center right'
	elif ( loc2 == 'upper outside' ):
		anchor = (0.5, 1)
		loc2 = 'lower center'
	elif ( loc2 == 'lower outside' ):
		anchor = (0.5, -0.2)  # depends on figure size, which is horrible
		loc2 = 'upper center'
	# end
	#print 'loc "' + loc2 + '", anchor', anchor
	
	# what's the best way to pass these things? this seems overly cumbersome
	if ( lines and labels ):
		pp.legend( lines, labels, loc=loc2, bbox_to_anchor=anchor, **kwargs )
	elif ( lines ):
		pp.legend( lines, loc=loc2, bbox_to_anchor=anchor, **kwargs )
	else:
		pp.legend( loc=loc2, bbox_to_anchor=anchor, **kwargs )
# end


# ------------------------------------------------------------
def resize( size=None, borders=None, space=None, pad=0.5, extra=None, figures=None ):
	'''
	Resizes and adjusts margins of matplotlib figures.
	
	size    is [ width, height ]
	
	borders is [ top, right, bottom, left ], as in HTML CSS
	
	space   is [ wspace, hspace ]
	
	If borders and space are not given, uses tight_layout()
	pad     is padding used in tight_layout()

	extra   is [ top, right, bottom, left ], extra space to add after tight_layout()
	
	figures is figure object, number, or list of numbers.
	        If None, the current figure (gcf) is used.
	'''
	import matplotlib.pyplot as pp
	
	if ( figures is None ):
		figures = [ pp.gcf() ]
	elif ( type(figures) in (int,numpy.int32) ):
		figures = [ pp.figure( figures ) ]
	elif ( type(figures) in (tuple,list) ):
		figures = map( pp.figure, figures )
	
	for fig in figures:
		if ( borders ):
			fig.subplots_adjust(
				top    = (1-borders[0]),
				right  = (1-borders[1]),
				bottom =    borders[2],
				left   =    borders[3],
			)
		if ( space ):
			fig.subplots_adjust(
				wspace = space[0],
				hspace = space[1]
			)
		if ( size ):
			fig.set_size_inches( size, forward=True )
		if ( not borders and not space ):
			fig.canvas.draw()
			fig.tight_layout( pad=pad )
			fig.canvas.draw()
		# end
		if ( extra ):
			t = fig.subplotpars.top
			l = fig.subplotpars.left
			b = fig.subplotpars.bottom
			r = fig.subplotpars.right
			fig.subplots_adjust(
				top    = t - extra[0],
				right  = r - extra[1],
				bottom = b + extra[2],
				left   = l + extra[3],
			)
			fig.canvas.draw()
	# end
	#pp.draw()
# end resize
