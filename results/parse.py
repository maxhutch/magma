#!/usr/bin/env python
#
# Parses MAGMA output and generates a python file,
# storing each run into its own numpy array.
#
# @author Mark Gates

import sys
import os
import re
import numpy


# --------------------
class data_t( object ):
	def __init__( self ):
		self.reset()
	
	def reset( self ):
		self.cmd  = ''
		self.name = ''
		self.opt  = ''
		self.rows = []
#end


# --------------------
# output one run as numpy array
def output( data ):
	if ( not data.rows ):
		return
	
	if ( not data.name ):
		data.name = 'unknown'
	
	# find maximum width of each column and make printf format
	n = len( data.rows[0] )
	maxwidths = [0] * n
	for row in data.rows:
		if ( len(row) != n ):
			print '# error: row has', len(widths), 'fields; first row had', n, 'fields'
			continue
		widths = map( len, row )
		for i in xrange( n ):
			maxwidths[i] = max( maxwidths[i], widths[i] )
	formats = map( lambda x: '%%%ds' % x, maxwidths )
	format = '\t[ ' + ',  '.join( formats ) + ' ],'
	
	name = data.name + data.opt
	
	# output table
	print '#', data.cmd
	print name, '= array(['
	for row in data.rows:
		#print format, row
		try:
			print format % tuple(row)
		except:
			print '# ERROR', format, row
	print '])\n'
	
	data.reset()
# end


# --------------------
# process one file
def process( filename ):
	data = data_t()
	warmup = 0
	
	print '# ------------------------------------------------------------'
	print '# file:', filename
	
	infile = open( filename )
	for line in infile:
		# look for header line
		m = re.search( r'^(?:numactl.*)?\./testing_(\w+)', line )
		if ( m ):
			if ( data.rows ):
				output( data )
			
			data.name = m.group(1)
			data.cmd  = line.strip()
			warmup = 2
			
			opt = ''
			m2 = re.search( r'-([LU])\b',       line )  # lower/upper
			if ( m2 ): opt += '_' + m2.group(1)
			m2 = re.search( r'-([UV][ASON])\b', line )  # svd U & V vectors
			if ( m2 ): opt += '_' + m2.group(1)
			m2 = re.search( r'-([JRL][NV])\b',  line )  # syev job vectors, geev right & left vectors
			if ( m2 ): opt += '_' + m2.group(1)
			data.opt  = opt
			continue;
		# end
		
		# look for usage line (in case no header line)
		m = re.search( r'Usage: ./testing_(\w+)', line )
		if ( m ):
			if ( data.rows ):
				output( data )
			data.name = m.group(1)
			warmup = 2
		# end
		
		# look for data lines
		# differentiating data lines from other output is not so easy.
		# look for lines containing numbers and excluding certain punctuation
		m  = re.search( r'\b\d+\.\d+\b', line )
		m2 = re.search( r'[%#:=/,]', line )
		if ( m and not m2 ):
			# remove () parens
			# convert --- and words (usually options like "S") to nan
			line2 = re.sub( r'[()]',          ' ',     line  )
			line2 = re.sub( r'\b[a-zA-Z]+\b', ' nan ', line2 )
			line2 = re.sub( r'\s---\s',       ' nan ', line2 )
			line2 = line2.strip()
			
			# gesvd has two job columns, usually the same, while gesdd has
			# only one job column. This eliminates 2nd job column for gesvd.
			if ( data.name == 'gesvd' ):
				line2 = re.sub( r'^( *nan) +nan', r'$1', line2 )
			
			fields = re.split( ' +', line2 )
			
			# verify that everything is numeric
			try:
				map( float, fields )
			except:
				print >>sys.stderr, 'ignoring:', line.strip(), '\n>       ', line2
				continue
			
			# skip warmup runs (N = 100 or 1000 in first two runs)
			if ( warmup > 0 ):
				warmup -= 1
				m = re.search( r'^ *([a-zA-Z]+ +)*1000?\b', line )
				if ( m ):
					continue
			
			data.rows.append( fields )
		# end
	# end
	output( data )
# end


# --------------------
print 'import numpy'
print 'from numpy import array, nan, inf'
print

if ( len(sys.argv) > 1 ):
	m = re.search( 'v?(\d+\.\d+\.\d+|trunk)/cuda(\d+\.\d+)-(\w+)/', sys.argv[1] )
	if ( m ):
		print "version = '%s'" % (m.group(1))
		print "cuda    = '%s'" % (m.group(2))
		print "device  = '%s'" % (m.group(3))
		print
	else:
		print >>sys.stderr, "Warning: no version information"
	# end
# end

for f in sys.argv[1:]:
	process( f )
