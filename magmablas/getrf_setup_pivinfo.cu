/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2013
       
       @author Azzam Haidar
       @author Tingxing Dong

*/
#include "common_magma.h"
#include "batched_kernel_param.h"



//=================================================================================================
//=================================================================================================
// AUXILIARY ROUTINE TO COMPUTE PIV FINAL DESTINATION FOR THE CURRENT STEP
//=================================================================================================
//=================================================================================================



//=================================================================================================
static __device__ void setup_pivinfo_devfunc(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb)
{
   int tid = threadIdx.x;   
   int nchunk = (m-1)/MAX_NTHREADS + 1;

    // initialize pivinfo (could be done in a separate kernel using multiple thread block
    for(int s =0 ; s < nchunk; s++)
    {
       if( (tid + s * MAX_NTHREADS < m) && (tid < MAX_NTHREADS) )   pivinfo[tid + s * MAX_NTHREADS] = tid + s * MAX_NTHREADS + 1;
    }
   __syncthreads();

   if(tid==0)
   {   
       int i, itsreplacement, mynewrowid;
       for(i=0; i<nb; i++){
           mynewrowid          = ipiv[i]-1; //-1 to get the index in C
           itsreplacement      = pivinfo[mynewrowid];
           pivinfo[mynewrowid] = pivinfo[i];
           pivinfo[i]          = itsreplacement;
       }
   }
}
//=================================================================================================
__global__ void setup_pivinfo_kernel_batched(magma_int_t **pivinfo_array, magma_int_t **ipiv_array, int m, int nb)
{
   int batchid = blockIdx.x;
   setup_pivinfo_devfunc(pivinfo_array[batchid], ipiv_array[batchid], m, nb);
}
//=================================================================================================

//=================================================================================================
__global__ void setup_pivinfo_kernel(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb)
{
   setup_pivinfo_devfunc(pivinfo, ipiv, m, nb);
}
//=================================================================================================





//TODO add description
//=================================================================================================
extern "C" void
setup_pivinfo_batched( magma_int_t **pivinfo_array, magma_int_t **ipiv_array, 
                         magma_int_t m, magma_int_t nb, 
                         magma_int_t batchCount,
                         magma_queue_t queue)
{
    if(nb == 0 ) return ;
    setup_pivinfo_kernel_batched<<<batchCount, min(m, MAX_NTHREADS), 0, queue>>>(pivinfo_array, ipiv_array, m, nb);
}



//=================================================================================================


//TODO add description
//=================================================================================================
extern "C" void
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, 
                 magma_int_t m, magma_int_t nb, 
                 magma_queue_t queue)
{
    if(nb == 0 ) return ;
    setup_pivinfo_kernel<<<1, min(m, MAX_NTHREADS), 0, queue>>>(pivinfo, ipiv, m, nb);
}
//=================================================================================================




//=================================================================================================
//=================================================================================================
// AUXILIARY ROUTINE TO ADJUST IPIV
//=================================================================================================
//=================================================================================================




//=================================================================================================
static __device__ void adjust_ipiv_devfunc(magma_int_t *ipiv, int m, int offset)
{
   int tid = threadIdx.x;
   if(tid < m)
   {
     ipiv[tid] += offset;
   }
}
//=================================================================================================
__global__ void adjust_ipiv_kernel_batched(magma_int_t **ipiv_array, int m, int offset)
{
   int batchid = blockIdx.x;
   adjust_ipiv_devfunc(ipiv_array[batchid], m, offset);
}
//=================================================================================================

//=================================================================================================
__global__ void adjust_ipiv_kernel(magma_int_t *ipiv, int m, int offset)
{
   adjust_ipiv_devfunc(ipiv, m, offset);
}
//=================================================================================================

//TODO add description
//=================================================================================================
extern "C" void
adjust_ipiv_batched( magma_int_t **ipiv_array, 
                         magma_int_t m, magma_int_t offset, 
                         magma_int_t batchCount, magma_queue_t queue)
{
    if(offset == 0 ) return ;
    if( m  > MAX_NTHREADS) 
    {
       printf(" adjust_ipiv_batched_q m=%d > %d, not supported \n", m, MAX_NTHREADS);
       return;
    }
    adjust_ipiv_kernel_batched<<<batchCount, m, 0, queue>>>(ipiv_array, m, offset);
}



//=================================================================================================





//TODO add description
//=================================================================================================
extern "C" void
adjust_ipiv( magma_int_t *ipiv, 
                 magma_int_t m, magma_int_t offset, 
                 magma_queue_t queue)
{
    if(offset == 0 ) return ;
    if( m  > 1024) 
    {
       printf(" adjust_ipiv_q m=%d > %d, not supported \n", m, MAX_NTHREADS);
       return;
    }
    adjust_ipiv_kernel<<<1, m, 0, queue>>>(ipiv, m, offset);
}











