#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

filename = "testing_results.txt"

# erase the file if it exists
f = open(filename, "w")
f.close()

# Add current directory to the path for subshells of this shell
# Allows the popen to find local files in both windows and unix
os.environ["PATH"] += ":."

print
print '---------------- LAPACK LIN Testing with MAGMA ----------------'
print
print '-- Detailed results are stored in', filename

dtypes = (
    ("s", "d", "c", "z"),
    ("Single", "Double", "Complex", "Double Complex"),
)
for dtype in range( len( dtypes[0] )):
    letter = dtypes[0][dtype]
    name   = dtypes[1][dtype]
    print
    print "------------------------- %s ------------------------" % name
    print
    sys.stdout.flush() # make sure progress of testing is shown
    f = open(filename, "a")
    test1 = os.popen("xlintst%s < %stest.in" % (letter, letter))
    for line in test1.readlines():
        f.write( line )
        if "passed"   in line : print line,
        if "failed"   in line : print "\n Failure =======>", line
        if "recorded" in line : print "\n ===>", line
    f.close()
