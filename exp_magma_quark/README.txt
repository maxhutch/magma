    -- MAGMA (version 1.6.1) -- 
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
magma_insert: The wrapper for any task insertion. e.g magma_insert_dgemm(...)
magma_task: The GPU routines called by the scheduler. e.g magma_task_dgemm(...)
schedule: where the scheduling take place.
src     : all available linear algebra routines (dgetrf_async, ...)
test    : testing directory
         
Test:
====
1. Compile magma
	cd <magma_dir>/trunk
	make
	
2. Edit make.exp.inc, set the peak performance of the (CPU, GPU) 
a) If you know the peak performance of the (CPU, GPU) you should specified them in the make.exp.inc and uncomment the corresponding lines beginning with MORE_INC in the make.exp.inc file.
b) Else if you know the performance of (cpu_dgemm, gpu_dgemm), you can use them as the peak performance of the (CPU, GPU) and then uncomment the corresponding lines beginning with MORE_INC in the make.exp.inc file.
c) Else if you don't know any of these, comment the entries CPU_PEAK and GPU_PEAK in the make.exp.inc. In that case, you will have to specify the parameter fraction_dcpu during the runtime.  
 
3. Compile magma_quark
	cd exp_magma_quark
	make
	
4. Test
	cd test
	./testing_dgetrf_async_gpu -N 10112 --nb 128 --nthread 16 --panel_nthread 4 --fraction 0.2
		 OR
	numactl --interleave=all ./testing_dgetrf_gpu_amc -N 10112 --nb 128 --nthread 16 --panel_nthread 4 --fraction_dcpu 0.2
	numactl --interleave=all ./testing_dgetrf_mgpu_amc -N 10112  --nb 128 --ngpu 4 --nthread 16 --panel_nthread 4 --fraction_dcpu 0.05	
	Arguments are:
      * nb: block size.
	  * nthread: total number of threads.
	  * panel_nthread: number of threads working on the panel.
	  * fraction_dcpu: this parameter may affect considerably the performance, it indicates the percentage of the matrix to schedule on the CPUs. It is recommanded to set the (CPU,GPU) peak in the make.inc file and let the system automatically compute it. This may affect considerably the performances. It becomes optional when you specify the (CPU_PEAK, GPU_PEAK) in the make.exp.inc file.
IMPORTANT NOTE: the code need to be ran with numactl --interleave=all if your are on NUMA machines.
