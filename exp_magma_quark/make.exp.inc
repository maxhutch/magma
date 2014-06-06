#
# CPU peak performance (for one cpu only)
# 
# Leave blank is unknown
# CPU_PEAK = 11.2
CPU_PEAK = 

#
# GPU peak performance (for one GPU only)
# 
# Leave blank is unknown
# GPU_PEAK = 504.0
GPU_PEAK = 

LIB += -L$(MAGMA_DIR)/lib -L$(MAGMA_DIR)/exp_magma_quark -lmagma_quark

#
# Quark
#
QUARKDIR ?= $(MAGMA_DIR)/quark
INC += -I$(QUARKDIR)
LIB += -L$(QUARKDIR) -lquark

INC += -I../include -I../schedule -I../control -I../core -I../debug -I../schedule

LIBMAGMA_QUARK = $(MAGMA_DIR)/exp_magma_quark/libmagma_quark.a

#
# 
#
INC += -I../../testing
LIB += -L../../testing -ltest