#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.6.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#
#	   @author: Simplice Donfack
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal
include make.exp.inc

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.o)

.PHONY: all clean cleanall

all: libquark libexp testing

libexp: 
	@echo ======================================== control
	( cd control  && $(MAKE) )
	@echo ======================================== debug
	( cd debug    && $(MAKE) )
	@echo ======================================== core
	( cd core     && $(MAKE) )
	@echo ======================================== schedule
	( cd schedule && $(MAKE) )
	@echo ======================================== magma_insert
	( cd magma_insert && $(MAKE) )
	@echo ======================================== magma_task
	( cd magma_task && $(MAKE) )
	@echo ======================================== src
	( cd src      && $(MAKE) )
	
libquark:
	@echo ======================================== quark
	( cd $(QUARKDIR)          && $(MAKE) CC=gcc)

testing:	
	@echo ======================================== test
	( cd test      && $(MAKE) )
	
clean:
	( cd control  && $(MAKE) clean )
	( cd debug    && $(MAKE) clean )
	( cd core     && $(MAKE) clean )
	( cd schedule && $(MAKE) clean )
	( cd src      && $(MAKE) clean )
	( cd test     && $(MAKE) clean )
	rm -f *.o *~ *.linkinfo

cleanall: clean
	rm -f $(LIBMAGMA)
