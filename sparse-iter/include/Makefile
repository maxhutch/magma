#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.7.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date September 2015
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/include
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

ZHDR := \
	magmasparse_z.h		\
	magmasparse_zc.h	\
	magmasparse_types.h	\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

.PHONY: all clean cleanall

all: $(ALLHDR)

clean:
	rm -f *~ *.mod

cleanall: clean
