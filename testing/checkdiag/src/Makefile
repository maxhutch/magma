#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.0.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date May 2016
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../../..
include $(MAGMA_DIR)/Makefile.internal

LIBNAME = ../lib/libcheck.a
lib: $(LIBNAME)
ALLOBJ=dcheck_eig.$(o_ext) dscala.$(o_ext) dscalde.$(o_ext) dstt21.$(o_ext) dsbt21.$(o_ext) dsyt21.$(o_ext) dlarfy.$(o_ext) \
    zcheck_eig.$(o_ext) zstt21.$(o_ext) zhbt21.$(o_ext) zhet21.$(o_ext) zlarfy.$(o_ext) checkdiag.$(o_ext)

$(LIBNAME): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@


#///////////////////////////////////////////////////////////////////////////////////////////////////

clean:
	rm -f *.$(o_ext) $(LIBNAME)

cleanall:
	rm -f *.$(o_ext) 
#///////////////////////////////////////////////////////////////////////////////////////////////////
