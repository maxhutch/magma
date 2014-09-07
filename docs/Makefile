all:
	doxygen Doxyfile

# Doxyfile-fast explicitly lists files in the SVN,
# excluding all the precision-generated files.
# It is a lot faster for checking doxygen warnings,
# but of course generates only a subset of the documentation.
# See output_err file for Doxygen warnings.
fast:
	-mv output_err old_output_err
	doxygen Doxyfile-fast

# Doxyfile-test is for local testing of specific files.
# Create Doxyfile-test as a local copy of Doxyfile-fast with INPUT as whatever
# files that you want to test, but do not add it to the SVN.
test:
	-mv output_err old_output_err
	doxygen Doxyfile-test
