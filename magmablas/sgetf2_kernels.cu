/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from zgetf2_kernels.cu normal z -> s, Fri Jan 30 19:00:10 2015
*/

#include "common_magma.h"
#include "magmablas.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"


#define PRECISION_s

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ float shared_data[];
extern __shared__ float sdata[];
extern __shared__ int int_sdata[];

/*
  routines in this file are used by sgetf2_batched.cu
*/

//////////////////////////////////////////////////////////////////////////////////////////

__device__ int 
isamax_devfunc(int length, const float *x, int incx, float *shared_x, int *shared_idx)
{

    int tx = threadIdx.x;
    float res;
    float  res1;
    int nchunk = (length-1)/zamax + 1;

    if( tx < zamax ){
        shared_x[tx]   = 0.0;
        shared_idx[tx] = tx;//-1;// -1 will crash the code in case matrix is singular, better is to put =tx and make check info at output
    }
    __syncthreads();

    for(int s =0 ; s < nchunk; s++)
    {
        if( (tx + s * zamax < length) && (tx < zamax) )
        {
            res = x[(tx + s * zamax) * incx];                   
            res1 = fabs(MAGMA_S_REAL(res)) + fabs(MAGMA_S_IMAG(res));
            
            if( res1  > shared_x[tx] )
            {
                shared_x[tx] = res1;
                shared_idx[tx] = tx + s * zamax;   
            }
           
        }
        __syncthreads();
    }

    if(length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
        magma_getidmax<zamax>(tx, shared_x, shared_idx);
    else
        magma_getidmax_n(min(zamax,length), tx, shared_x, shared_idx);
    return shared_idx[0];

}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void 
isamax_kernel_batched(int length, int chunk, float **x_array, int incx, 
                   int step, int lda, magma_int_t** ipiv_array, magma_int_t *info_array, int gbstep)
{

    float *x_start = x_array[blockIdx.z];
    const float *x = &(x_start[step + step * lda]); 

    magma_int_t *ipiv = ipiv_array[blockIdx.z];
    int tx = threadIdx.x;

    float *shared_x = sdata;     
    int *shared_idx = (int*)(shared_x + zamax);
    
    isamax_devfunc(length, x, incx, shared_x, shared_idx);
    
    if(tx == 0){
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if(shared_x[0] == MAGMA_D_ZERO){
            info_array[blockIdx.z] = shared_idx[0] + step + gbstep + 1;
        }
    }


}

////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void
tree_isamax_kernel_batched(int length, float **x_array, int incx, 
                   int step, int lda, magma_int_t** ipiv_array, magma_int_t *info_array, int gbstep, 
                   float** data_pool_array, magma_int_t** id_pool_array)
{

    float *x_start = x_array[blockIdx.z];
    const float *x = &(x_start[step + step * lda]); 

    float *data_pool = data_pool_array[blockIdx.z];
    magma_int_t *id_pool = id_pool_array[blockIdx.z];

    magma_int_t *ipiv = ipiv_array[blockIdx.z];
    int tx = threadIdx.x;
    int local_max_id;

    __shared__ float shared_x[zamax];
    __shared__ int    shared_idx[zamax];
    
    x += zamax * blockIdx.x * incx;

    isamax_devfunc(min(zamax, length-blockIdx.x * zamax), x, incx, shared_x, shared_idx);
  
    if(tx ==0) 
    {
        local_max_id = shared_idx[0] + zamax * blockIdx.x; // add the offset

        if(gridDim.x == 1) 
        {
            ipiv[step]  = local_max_id + step + 1; // Fortran Indexing
            if(shared_x[0] == MAGMA_D_ZERO)
                info_array[blockIdx.z] = local_max_id + step + gbstep + 1;
        }
        else
        {
            // put each thread block local max and its index in workspace
            data_pool[blockIdx.x] = shared_x[0]; 
            id_pool[blockIdx.x] = local_max_id;
        }


    } 


}



__global__ void
tree_isamax_kernel2_batched(int n, int step,  magma_int_t** ipiv_array, magma_int_t *info_array, int gbstep, float** data_pool_array, magma_int_t** id_pool_array)
{
    __shared__ float shared_x[zamax];
    __shared__ int    shared_idx[zamax];

    magma_int_t *ipiv = ipiv_array[blockIdx.z];

    float *data_pool = data_pool_array[blockIdx.z];
    magma_int_t *id_pool = id_pool_array[blockIdx.z];


    int tx = threadIdx.x;

    //read data
    if( tx < n)
    {
        shared_x[tx] = data_pool[tx];
        shared_idx[tx] = id_pool[tx]; 
    } 
    else
    {
        shared_x[tx] = 0.0;
        shared_idx[tx] = -2; 
    }
 
    __syncthreads();
    
    // compute local result inside each thread block
    magma_getidmax<zamax>(tx, shared_x, shared_idx);


    if(tx == 0 ) 
    {
            ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
            if(shared_x[0] == MAGMA_D_ZERO)
                info_array[blockIdx.z] = shared_idx[0] + step + gbstep + 1;
    } 

}




magma_int_t magma_isamax_lg_batched(magma_int_t length, float **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)

{
    if(length == 1) return 0;
    if(incx < 0) return 1;
    
    float* data_pool; 
    magma_int_t* id_pool;

    float** data_pool_array = NULL;  
    magma_int_t** id_pool_array = NULL;
 
    magma_int_t num_blocks = (length-1)/(zamax) + 1;

    // creat pools(data and index) to store the result of each thread blocks
    magma_smalloc(&data_pool, num_blocks * batchCount);
    magma_imalloc(&id_pool,   num_blocks * batchCount);
 
    magma_malloc((void**)&data_pool_array, batchCount * sizeof(*data_pool_array));
    magma_malloc((void**)&id_pool_array, batchCount * sizeof(*id_pool_array));

#if defined(PRECISION_z) || defined(PRECISION_d)
    dset_pointer(data_pool_array, data_pool, 1, 0, 0, num_blocks, batchCount, queue);
#else
    sset_pointer(data_pool_array, data_pool, 1, 0, 0, num_blocks, batchCount, queue);
#endif 

    set_ipointer(id_pool_array, id_pool, 1, 0, 0, num_blocks, batchCount, queue);


    if( num_blocks > zamax) 
    {
        printf("length(=%d), num_blocks(=%d) is too big > zamax(=%d), the second layer reduction can not be launched, Plz incread zamax \n", length, num_blocks, zamax);
    } 
    else
    {
        // first level tree reduction
        dim3 grid(num_blocks, 1, batchCount);

        tree_isamax_kernel_batched<<<grid, zamax, 0, queue>>>(length, x_array, incx, step, lda, ipiv_array, info_array, gbstep, data_pool_array, id_pool_array);

        if( num_blocks > 1)
        {
            // second level tree reduction
            dim3 grid2(1, 1, batchCount);
            tree_isamax_kernel2_batched<<<grid2, zamax, 0, queue>>>(num_blocks, step,  ipiv_array, info_array, gbstep, data_pool_array, id_pool_array);
        }
    }


    magma_free(data_pool);
    magma_free(id_pool);

    magma_free(data_pool_array);
    magma_free(id_pool_array);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C"
magma_int_t magma_isamax_batched(magma_int_t length, 
        float **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
  
    if(length == 0 ) return 0;

#if 1
        dim3 grid(1, 1, batchCount);
        int chunk = (length-1)/zamax + 1;
        isamax_kernel_batched<<< grid, zamax, zamax * (sizeof(float) + sizeof(int)), queue >>>
                      (length, chunk, x_array, incx, step, lda, ipiv_array, info_array, gbstep);

#else
    // the magma_isamax_lg_batched is faster but when cuda launch it as 2 kernels the white space time between these 2 kernels and the next kernel is larger than using the isamax_kernel for that today we are using only isamax_kernel
    if( length <= 10 * zamax )
    {  
        dim3 grid(1, 1, batchCount);
        int chunk = (length-1)/zamax + 1;
        isamax_kernel_batched<<< grid, zamax, zamax * (sizeof(float) + sizeof(magma_int_t)), queue >>>
                      (length, chunk, x_array, incx, step, lda, ipiv_array, info_array, gbstep);

    }
    else
    {
        magma_isamax_lg_batched(length, x_array, incx, step, lda, ipiv_array, info_array, gbstep, batchCount);
    }
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


__global__
void sswap_kernel_batched(magma_int_t n, float **x_array, magma_int_t incx, magma_int_t step, magma_int_t** ipiv_array)
{

    float *x = x_array[blockIdx.z];
    magma_int_t *ipiv = ipiv_array[blockIdx.z];

    __shared__ int jp;
    
    if(threadIdx.x == 0) 
    {
      jp = ipiv[step] - 1;
      //if(blockIdx.z == 1) printf("jp=%d", jp);
    } 
    __syncthreads();
 
    if(jp == step)  return; // no pivot

    int id = threadIdx.x;

    if (id < n) {
        float tmp = x[jp + incx*id];
        x[jp + incx*id] = x[step + incx*id];
        x[step + incx*id] = tmp;
    }

}
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
magma_int_t magma_sswap_batched(magma_int_t n, float **x_array, magma_int_t incx, magma_int_t step, 
                 magma_int_t** ipiv_array, magma_int_t batchCount, magma_queue_t queue)
{
/*
    sswap two row: (ipiv[step]-1)th and jth
*/
    if( n  > MAX_NTHREADS) 
    {
       printf("magma_sswap_batched nb=%d, > %d, not supported \n",n, MAX_NTHREADS);
       return -15;
    }
    dim3 grid(1,1, batchCount);
    sswap_kernel_batched<<< grid, n, 0, queue >>>(n, x_array, incx, step, ipiv_array);
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void sscal_sger_kernel_batched(int m, int n, int step, float **dA_array, int lda, magma_int_t *info_array, int gbstep)
{

    // checkinfo to avoid computation of the singular matrix
    if(info_array[blockIdx.z] != 0 ) return;

    float *A_start = dA_array[blockIdx.z];
    float *A = &(A_start[step + step * lda]); 
    float *shared_y = shared_data;

    int tx  = threadIdx.x;
    int gbidx = blockIdx.x*MAX_NTHREADS + threadIdx.x;

    if (tx < n) {
        shared_y[tx] = A[lda * tx];
    }
    __syncthreads();
    if(shared_y[0] == MAGMA_S_ZERO) {
        info_array[blockIdx.z] = step + gbstep + 1; 
        return;
    }

    if (gbidx < m && gbidx > 0) {
        float reg = MAGMA_S_ZERO;
        reg = A[gbidx];
        reg *= MAGMA_S_DIV(MAGMA_S_ONE, shared_y[0]);
        A[gbidx] = reg;
        #pragma unroll
        for(int i=1; i < n; i++) {
            //A[gbidx + i*lda] = A[gbidx + i*lda] - shared_y[i] * reg;//cuda give wrong results with this one
            //A[gbidx + i*lda] -= shared_y[i] * reg; //cuda give wrong results with this one
            A[gbidx + i*lda] += (MAGMA_S_NEG_ONE) * shared_y[i] * reg;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
magma_int_t magma_sscal_sger_batched(magma_int_t m, magma_int_t n, magma_int_t step,
                                      float **dA_array, magma_int_t lda,
                                      magma_int_t *info_array, magma_int_t gbstep, 
                                      magma_int_t batchCount, magma_queue_t queue)
{
/*

    Specialized kernel which merged sscal and sger the two kernels
    1) sscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a sger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where 
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);

*/
    if( n == 0) return 0;
    if( n  > MAX_NTHREADS) 
    {
       printf("magma_sscal_sger_batched nb=%d, > %d, not supported \n",n, MAX_NTHREADS);
       return -15;
    }

    int nchunk = (m-1)/MAX_NTHREADS + 1;
    size_t shared_size = sizeof(float)*(n);
    dim3 grid(nchunk, 1, batchCount);

    sscal_sger_kernel_batched<<< grid, min(m, MAX_NTHREADS), shared_size, queue>>>(m, n, step, dA_array, lda, info_array, gbstep);
    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void sgetf2trsm_kernel_batched(int ib, int n, float **dA_array, int step, int lda)
{
        /*
           this kernel does the safe nonblocked TRSM operation
           B = A^-1 * B
         */ 

    float *A_start = dA_array[blockIdx.z];
    float *A = &(A_start[step + step * lda]); 
    float *B = &(A_start[step + (step+ib) * lda]); 
    float *shared_a = shared_data;
    float *shared_b = shared_data+ib*ib;

    int tid = threadIdx.x;
    int i,d;


    // Read A and B at the same time to the shared memory (shared_a shared_b)
    // note that shared_b = shared_a+ib*ib so its contiguous 
    // I can make it in one loop reading  
    if ( tid < ib) {
        #pragma unroll
        for( i=0; i < n+ib; i++) {
             shared_a[tid + i*ib] = A[tid + i*lda];
        }
    }
    __syncthreads();

    if (tid < n) {
        #pragma unroll
        for( d=0;  d<ib-1; d++) {
            for( i=d+1; i<ib; i++) {
                shared_b[i+tid*ib] += (MAGMA_S_NEG_ONE) * shared_a[i+d*ib] * shared_b[d+tid*ib];
            }
        }
    }
    __syncthreads();

    // write back B
    if ( tid < ib) {
        #pragma unroll
        for( i=0; i < n; i++) {
              B[tid + i*lda] = shared_b[tid + i*ib];
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magma_sgetf2trsm_batched(magma_int_t ib, magma_int_t n, float **dA_array,  magma_int_t step, magma_int_t lda,
                       magma_int_t batchCount, magma_queue_t queue)
{
/*

*/
    if( n == 0 || ib == 0 ) return;
    size_t shared_size = sizeof(float)*(ib*(ib+n));

    // TODO TODO TODO
    if( shared_size >  (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 46K leaving 2K for extra
    {
        printf("kernel_sgetf2trsm error out of shared memory \n");
        return;
    }

    dim3 grid(1, 1, batchCount);
    sgetf2trsm_kernel_batched<<< grid, max(n,ib), shared_size, queue>>>(ib, n, dA_array, step, lda);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ void 
zupdate_device(int m, int step, float* x, int ldx,  float *A, int lda)
{

    int tid = threadIdx.x;
    int nchunk = (m-1)/MAX_NTHREADS + 1;    
    int indx;
    //float reg = MAGMA_S_ZERO;

    // update the current column by all the previous one
    #pragma unroll
    for(int i=0; i < step; i++) {
        for(int s=0 ; s < nchunk; s++)
        {
            indx = tid + s * MAX_NTHREADS;
            if ( indx > i  && indx < m ) {
                A[indx] -=  A[i] * x[indx + i*ldx];
                //printf("         @ step %d tid %d updating x[tid]*y[i]=A %5.3f %5.3f = %5.3f  at i %d \n", step, tid, x[tid + i*ldx], A[i], A[tid],i);
            }
        }
        __syncthreads();
    }

    //printf("         @ step %d tid %d adding %5.3f to A %5.3f make it %5.3f\n",step,tid,-reg,A[tid],A[tid]-reg);

}

////////////////////////////////////////////////////////////////////////////////////////////////////
static __device__ void 
sscal5_device(int m, float* x, float alpha)
{
    int tid = threadIdx.x;
    int nchunk = (m-1)/MAX_NTHREADS + 1;    

    for(int s=0 ; s < nchunk; s++)
    {
        if( (tid + s * MAX_NTHREADS) < m ) {
            #if 0
            x[tid + s * MAX_NTHREADS] *= MAGMA_S_DIV(MAGMA_S_ONE, alpha);
            #else
            x[tid + s * MAX_NTHREADS] = x[tid + s * MAX_NTHREADS]/alpha;
            #endif
        }
    }
    __syncthreads();
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void 
zcomputecolumn_kernel_shared_batched(int m, int paneloffset, int step, float **dA_array, int lda, magma_int_t **ipiv_array, magma_int_t *info_array, int gbstep)
{
    int gboff = paneloffset+step;
    magma_int_t *ipiv                   = ipiv_array[blockIdx.z];
    float *A_start = dA_array[blockIdx.z];
    float *A0j     = &(A_start[paneloffset + (paneloffset+step) * lda]); 
    float *A00     = &(A_start[paneloffset + paneloffset * lda]); 

    float *shared_A = shared_data;
    __shared__ float  shared_x[zamax];
    __shared__ int     shared_idx[zamax];
    __shared__ float alpha;
    int tid = threadIdx.x;

    // checkinfo to avoid computation of the singular matrix
    if(info_array[blockIdx.z] != 0 ) return;


    int nchunk = (m-1)/MAX_NTHREADS + 1;    
    // read the current column from dev to shared memory
    for(int s=0 ; s < nchunk; s++)
    {
        if( (tid + s * MAX_NTHREADS) < m ) shared_A[tid + s * MAX_NTHREADS] = A0j[tid + s * MAX_NTHREADS];
    }
    __syncthreads();

    // update this column
    if( step > 0 ){
        zupdate_device( m, step, A00, lda, shared_A, 1);
        __syncthreads();
    }

    // if( tid < (m-step) ) // DO NO TPUT THE IF CONDITION HERE SINCE isamax_devfunc HAS __syncthreads INSIDE. 
    // So let all htreads call this routine it will handle correctly based on the size
    // note that isamax need only 128 threads, s
    isamax_devfunc(m-step, shared_A+step, 1, shared_x, shared_idx);
    if(tid == 0){
        ipiv[gboff]  = shared_idx[0] + gboff + 1; // Fortran Indexing
        alpha = shared_A[shared_idx[0]+step];
        //printf("@ step %d ipiv=%d where gboff=%d  shared_idx %d alpha %5.3f \n",step,ipiv[gboff],gboff,shared_idx[0],alpha);
        if(shared_x[0] == MAGMA_D_ZERO){
            info_array[blockIdx.z] = shared_idx[0] + gboff + gbstep + 1;
        }
    }
    __syncthreads();
    if(shared_x[0] == MAGMA_D_ZERO) return;
    __syncthreads();

    // DO NO PUT THE IF CONDITION HERE SINCE isamax_devfunc HAS __syncthreads INSIDE.
    sscal5_device( m-step, shared_A+step, alpha);

    // put back the pivot that has been scaled with itself menaing =1 
    if(tid == 0)  shared_A[shared_idx[0] + step] = alpha;
    __syncthreads();

    // write back from shared to dev memory
    for(int s=0 ; s < nchunk; s++)
    {
        if( (tid + s * MAX_NTHREADS) < m )
        {
            A0j[tid + s * MAX_NTHREADS] = shared_A[tid + s * MAX_NTHREADS];
            //printf("@ step %d tid %d updating A=x*alpha after A= %5.3f\n",step,tid,shared_A[tid]);
        }            
    }
    __syncthreads();

}

////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
magma_int_t magma_scomputecolumn_batched(magma_int_t m, magma_int_t paneloffset, magma_int_t step, 
                                        float **dA_array,  magma_int_t lda,
                                        magma_int_t **ipiv_array, 
                                        magma_int_t *info_array, magma_int_t gbstep, 
                                        magma_int_t batchCount, magma_queue_t queue)
{
/*

    Specialized kernel which merged sscal and sger the two kernels
    1) sscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a sger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where 
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);

*/
    if( m == 0) return 0;

    size_t all_shmem_size = zamax*(sizeof(float)+sizeof(int)) + (m+2)*sizeof(float);
    if( all_shmem_size >  (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 44K leaving 4K for extra
    {
        printf("magma_scomputecolumn_batched error out of shared memory \n");
        return -20;
    }

    size_t shared_size = sizeof(float)*m;
    dim3 grid(1, 1, batchCount);
    zcomputecolumn_kernel_shared_batched<<< grid, min(m, MAX_NTHREADS), shared_size, queue>>>(m, paneloffset, step, dA_array, lda, ipiv_array, info_array, gbstep);

    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

