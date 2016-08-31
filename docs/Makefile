all:
	cd .. && $(MAKE) generate
	doxygen Doxyfile
	@echo "================================================================================"
	./summarize_errors.pl output_err

# Doxyfile-fast explicitly lists files in the SVN,
# excluding all the precision-generated files.
# It is a lot faster for checking doxygen warnings,
# but of course generates only a subset of the documentation.
# See output_err file for Doxygen warnings.
fast:
	cd .. && $(MAKE) generate
	doxygen Doxyfile-fast
	@echo "================================================================================"
	./summarize_errors.pl output_err_fast

# Doxyfile-test is for local testing of specific files.
# Create Doxyfile-test as a local copy of Doxyfile-fast with INPUT as whatever
# files that you want to test, but do not add it to the SVN.
test:
	cd .. && $(MAKE) generate
	doxygen Doxyfile-test
	@echo "================================================================================"
	./summarize_errors.pl output_err_test
