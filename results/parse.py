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
		self.cmd  = []
		self.name = ''
		self.name_usage = ''
		self.rows = []
#end


# --------------------
# output one run as numpy array
def output( data ):
	if ( not data.rows ):
		return
	
	if ( not data.name ):
		data.name = data.name_usage
	if ( not data.name ):
		data.name = 'unknown'
	
	# find maximum width of each column and make printf format
	n = len( data.rows[0] )
	maxwidths = [0] * n
	for row in data.rows:
		if ( len(row) != n ):
			print '# error: row has', len(row), 'fields; first row had', n, 'fields'
			continue
		widths = map( len, row )
		for i in xrange( n ):
			maxwidths[i] = max( maxwidths[i], widths[i] )
	formats = map( lambda x: '%%%ds' % x, maxwidths )
	format = '\t[ ' + ',  '.join( formats ) + ' ],'
	
	# output table
	for cmd in data.cmd:
		print '#', cmd
	print data.name, '= array(['
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
	warmup = 0
	
	print '# ------------------------------------------------------------'
	print '# file:', filename
	
	data   = data_t()
	keys   = []
	tables = {}
	
	infile = open( filename )
	for line in infile:
		# look for header line
		m = re.search( r'^(?:numactl.*)?testing_(\w+)', line )
		if ( m ):
			name = m.group(1)
			m2 = re.search( r'-([LU])\b',       line )  # lower/upper
			if ( m2 ): name += '_' + m2.group(1)
			m2 = re.search( r'-([UV][ASON])\b', line )  # svd U & V vectors
			if ( m2 ): name += '_' + m2.group(1)
			m2 = re.search( r'-([JRL][NV])\b',  line )  # syev job vectors, geev right & left vectors
			if ( m2 ): name += '_' + m2.group(1)
			
			# code repeated below
			if ( name in keys ):
				data = tables[name]
			else:
				data = data_t()
				data.name = name
				keys.append( name )
				tables[name] = data
			# end
			
			data.cmd.append( line.strip() )
			warmup = 2
			continue
		# end
		
		# look for usage line (in case no header line)
		m = re.search( r'Usage: ./testing_(\w+)', line )
		if ( m ):
			name = m.group(1)
			if ( data.rows ):
				# new table with no header
				# code repeated above
				if ( name in keys ):
					data = tables[name]
				else:
					data = data_t()
					data.name = name
					keys.append( name )
					tables[name] = data
				# end
				
				data.cmd.append( line.strip() )
				warmup = 2
				continue
			else:
				# table had header
				data.name_usage = name
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
			if ( re.search( 'gesvd', data.name )):
				line2 = re.sub( r'^( *nan) +nan', r'\1', line2 )
			
			fields = re.split( ' +', line2 )
			
			# verify that everything is numeric
			try:
				map( float, fields )
			except:
				print >>sys.stderr, 'ignoring:', line.strip()  #, '\n>       ', line2
				continue
			
			# skip warmup runs (N = 123, 1234 in first two runs)
			if ( warmup > 0 ):
				warmup -= 1
				m = re.search( r'^ *([a-zA-Z]+ +)*(1000?|1234?)\b', line )
				if ( m ):
					continue
			
			# for gesvd, skip second field, jobv
			# this makes it match gesdd, which has only job, not jobu and jobv
			if ( data.name[1:] == 'gesvd' ):
				fields = fields[0:1] + fields[2:]
			
			data.rows.append( fields )
		# end
	# end
	for key in keys:
		output( tables[key] )
# end


# --------------------
print 'import numpy'
print 'from numpy import array, nan, inf'
print

if ( len(sys.argv) > 1 ):
	m = re.search( 'v?(\d+\.\d+\.\d+|trunk)/cuda(\d+\.\d+)-(.*)/', sys.argv[1] )
	if ( m ):
		print "version = '%s'" % (m.group(1))
		print "cuda    = '%s'" % (m.group(2))
		print "device  = '%s'" % (m.group(3))
		print "cpu     = 'unknown'"
		print
	else:
		print "version = 'unknown'"
		print "cuda    = 'unknown'"
		print "device  = 'unknown'"
		print "cpu     = 'unknown'"
		print
		print >>sys.stderr, "\nWarning: no version information\n"
	# end
# end

for f in sys.argv[1:]:
	process( f )
