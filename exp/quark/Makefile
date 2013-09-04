include ../../make.inc


LIBQUARK = lib/libquark.so
ifeq (${ARCH}, ar)
	LIBQUARK = lib/libquark.a
endif

all: 	${LIBQUARK}

lib: 	${LIBQUARK}

clean:
	rm -f *.o *~

cleanall: clean
	rm -f ${LIBQUARK}

.c.o:
	$(CC) $(OPTS) -Iinclude $(INC) -c $< -o $@

${LIBQUARK}: icl_hash.o icl_list.o quarkos.o quark.o
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

.PHONY: clean

