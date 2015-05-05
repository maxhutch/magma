#!/bin/sh
#
# Finds doxygen groups that are in use,  sorts & puts in file "ingroup"
# Finds doxygen groups that are defined, sorts & puts in file "defgroup"
# Doing
#     diff ingroup defgroup
# provides an easy way to see what groups are used vs. defined.
#
# @author Mark Gates

grep -h '@ingroup' ../*/*.{h,c,cu,cpp} ../sparse-iter/*/*.{h,cu,cpp} | \
	perl -pe 's/^ *\*//;  s/^ +//;  s/\@ingroup/\@group/;' | \
	sort --unique > ingroup

# only take groups indented to 4th level --
# other groups are parents that shouldn't have functions directly inside them.
grep -h '^            @defgroup' doxygen-modules.h | \
	perl -pe 's/^( *)\@defgroup (\w+).*/\@group $2/;' | \
	sort > defgroup

opendiff ingroup defgroup
