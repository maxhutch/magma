    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       Sept 2013 
 
       @author: Simplice Donfack
	   @contact: {sdonfack, tomov}@eecs.utk.edu 
	   
Notice:
======
exp_magma_quark is a temporary name. We are still looking for a better name, so if you have a better suggestion, just let us know.

Purpose:
========
exp_magma_quark proposes a new scheduling strategy for hybrid CPU/GPU computations.
By default it uses quark, but any scheduler can be used by adapting very easily the wrapper file schedule_wrap_quark.h to schedule_wrap_<your_scheduler>.h.
The main idea behind the implementation :
	1. Initially distribute work between CPUs and GPUs based respectively on their corresponding peak performance. This in order to improve load balancing.
	2. Dynamically schedule tasks between CPUs and GPUs using a dynamic scheduler as Quark.
	3. During the runtime, move some data from the GPUs to the CPUs to balance work.

For more details: see http://web.eecs.utk.edu/~library/TechReports/2013/ut-cs-13-713.pdf 

Directory:
=========
control : auxiliary files, indirectly linked with linear algebra routines.
core    : cpu routines.
include : global header files.
schedule: where the scheduling take place.
src     : all available linear algebra routines (dgetrf_async, ...)
test    : testing directory
         
Test:
====
1. Compile magma
	cd <magma_dir>/trunk
	make
	
2. Compile magma_quark
	cd exp_magma_quark
	make
	
3. Test
	cd test
	./testing_dgetrf_async_gpu -N 10112 --nb 128 --nthread 16 --panel_nthread 4 --fraction 0.2
		 OR
	numactl --interleave=all ./testing_dgetrf_async_gpu -N 10112 --nb 128 --nthread 16 --panel_nthread 4 --fraction 0.2 
	Arguments are:
      * nb: block size.
	  * nthread: total number of threads.
	  * panel_nthread: number of threads working on the panel.
	  * fraction: the percentage of the matrix to allocate to the CPU. This may affect considerably the performances. A model to determine the parameter is in preparation.
IMPORTANT NOTE: the code need to be ran with numactl --interleave=all if your are on NUMA machines.