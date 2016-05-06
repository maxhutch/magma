/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from magmablas/zgetf2_kernels.cu normal z -> s, Mon May  2 23:30:41 2016
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"

/*

    Purpose
    -------
    These are internal routines that might have many assumption.
    They are used in sgetf2_batched.cpp   
    No documentation is available today.

    @ingroup magma_sgesv_aux

*/


#define PRECISION_s

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ float shared_data[];
extern __shared__ float sdata[];
extern __shared__ int int_sdata[];

//////////////////////////////////////////////////////////////////////////////////////////

__device__ int 
isamax_devfunc(int length, const float *x, int incx, float *shared_x, int *shared_idx)
{
    int tx = threadIdx.x;
    float res;
    float  res1;
    int nchunk = magma_ceildiv( length, zamax );

    if ( tx < zamax ) {
        shared_x[tx]   = 0.0;
        shared_idx[tx] = tx; //-1; // -1 will crash the code in case matrix is singular, better is to put =tx and make check info at output
    }
    __syncthreads();

    for (int s =0; s < nchunk; s++)
    {
        if ( (tx + s * zamax < length) && (tx < zamax) )
        {
            res = x[(tx + s * zamax) * incx];                   
            res1 = fabs(MAGMA_S_REAL(res)) + fabs(MAGMA_S_IMAG(res));
            
            if ( res1  > shared_x[tx] )
            {
                shared_x[tx] = res1;
                shared_idx[tx] = tx + s * zamax;   
            }
        }
        __syncthreads();
    }

    if (length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
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
    
    if (tx == 0) {
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if (shared_x[0] == MAGMA_D_ZERO) {
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
  
    if (tx == 0) 
    {
        local_max_id = shared_idx[0] + zamax * blockIdx.x; // add the offset

        if (gridDim.x == 1) 
        {
            ipiv[step]  = local_max_id + step + 1; // Fortran Indexing
            if (shared_x[0] == MAGMA_D_ZERO)
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
    if ( tx < n)
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


    if (tx == 0 ) 
    {
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if (shared_x[0] == MAGMA_D_ZERO)
            info_array[blockIdx.z] = shared_idx[0] + step + gbstep + 1;
    } 
}




magma_int_t magma_isamax_lg_batched(magma_int_t length, float **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)

{
    if (length == 1) return 0;
    if (incx < 0) return 1;
    
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
    magma_sset_pointer( data_pool_array, data_pool, 1, 0, 0, num_blocks, batchCount, queue );
#else
    magma_sset_pointer( data_pool_array, data_pool, 1, 0, 0, num_blocks, batchCount, queue );
#endif 

    magma_iset_pointer( id_pool_array, id_pool, 1, 0, 0, num_blocks, batchCount, queue );


    if ( num_blocks > zamax) 
    {
        fprintf( stderr, "%s: length(=%d), num_blocks(=%d) is too big > zamax(=%d), the second layer reduction can not be launched, Plz incread zamax\n",
                 __func__, int(length), int(num_blocks), int(zamax));
    }
    else
    {
        // first level tree reduction
        dim3 grid(num_blocks, 1, batchCount);
        dim3 threads(zamax, 1, 1);

        tree_isamax_kernel_batched
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (length, x_array, incx, step, lda, ipiv_array, info_array, gbstep, data_pool_array, id_pool_array);

        if ( num_blocks > 1)
        {
            // second level tree reduction
            dim3 grid2(1, 1, batchCount);
            tree_isamax_kernel2_batched
                <<< grid2, threads, 0, queue->cuda_stream() >>>
                (num_blocks, step,  ipiv_array, info_array, gbstep, data_pool_array, id_pool_array);
        }
    }


    magma_free(data_pool);
    magma_free(id_pool);

    magma_free(data_pool_array);
    magma_free(id_pool_array);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------

    ISAMAX find the index of max absolute value of elements in x and store the index in ipiv 

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    length       INTEGER
            On entry, length specifies the size of vector x. length >= 0.


    @param[in]
    x_array     Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension


    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            the offset of ipiv

    @param[in]
    lda    INTEGER
            The leading dimension of each array A, internal use to find the starting position of x.

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep    INTEGER
            the offset of info, internal use

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sgesv_aux
    ********************************************************************/

extern "C"
magma_int_t magma_isamax_batched(magma_int_t length, 
                                 float **x_array, magma_int_t incx, 
                                 magma_int_t step,  magma_int_t lda,
                                 magma_int_t** ipiv_array, magma_int_t *info_array, 
                                 magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    dim3 grid(1, 1, batchCount);
    dim3 threads(zamax, 1, 1);

#if 1

    int chunk = magma_ceildiv( length, zamax );
    isamax_kernel_batched<<< grid, threads, zamax * (sizeof(float) + sizeof(int)), queue->cuda_stream() >>>
        (length, chunk, x_array, incx, step, lda, ipiv_array, info_array, gbstep);

#else
    // the magma_isamax_lg_batched is faster but when cuda launch it as 2 kernels the white space time between these 2 kernels and the next kernel is larger than using the isamax_kernel for that today we are using only isamax_kernel
    if ( length <= 10 * zamax )
    {  
        int chunk = magma_ceildiv( length, zamax );
        isamax_kernel_batched<<< grid, threads, zamax * (sizeof(float) + sizeof(magma_int_t)), queue->cuda_stream() >>>
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
    
    if (threadIdx.x == 0) 
    {
        jp = ipiv[step] - 1;
        //if (blockIdx.z == 1) printf("jp=%d", jp);
    } 
    __syncthreads();
 
    if (jp == step)  return; // no pivot

    int id = threadIdx.x;

    if (id < n) {
        float tmp = x[jp + incx*id];
        x[jp + incx*id] = x[step + incx*id];
        x[step + incx*id] = tmp;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------

    sswap two row in x.  index (ipiv[step]-1)-th and index step -th

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            On entry, n specifies the size of vector x. n >= 0.


    @param[in]
    x_array     Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension


    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sgesv_aux
    ********************************************************************/

extern "C"
magma_int_t magma_sswap_batched(magma_int_t n, float **x_array, magma_int_t incx, 
                                magma_int_t step, magma_int_t** ipiv_array, 
                                magma_int_t batchCount, magma_queue_t queue)
{
    /*
    sswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS) 
    {
        fprintf( stderr, "%s nb=%d > %d, not supported\n", __func__, int(n), int(MAX_NTHREADS) );
        return -15;
    }
    dim3 grid(1,1, batchCount);
    dim3 threads(zamax, 1, 1);

    sswap_kernel_batched
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, x_array, incx, step, ipiv_array);
    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void sscal_sger_kernel_batched(int m, int n, int step, float **dA_array, int lda, magma_int_t *info_array, int gbstep)
{
    // checkinfo to avoid computation of the singular matrix
    if (info_array[blockIdx.z] != 0 ) return;

    float *A_start = dA_array[blockIdx.z];
    float *A = &(A_start[step + step * lda]); 
    float *shared_y = shared_data;

    int tx  = threadIdx.x;
    int gbidx = blockIdx.x*MAX_NTHREADS + threadIdx.x;

    if (tx < n) {
        shared_y[tx] = A[lda * tx];
    }
    __syncthreads();
    if (shared_y[0] == MAGMA_S_ZERO) {
        info_array[blockIdx.z] = step + gbstep + 1; 
        return;
    }

    if (gbidx < m && gbidx > 0) {
        float reg = MAGMA_S_ZERO;
        reg = A[gbidx];
        reg *= MAGMA_S_DIV(MAGMA_S_ONE, shared_y[0]);
        A[gbidx] = reg;
        #pragma unroll
        for (int i=1; i < n; i++) {
            //A[gbidx + i*lda] = A[gbidx + i*lda] - shared_y[i] * reg; //cuda give wrong results with this one
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
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) 
    {
        fprintf( stderr, "%s nb=%d, > %d, not supported\n", __func__, int(n), int(MAX_NTHREADS) );
        return -15;
    }

    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    size_t shared_size = sizeof(float)*(n);

    dim3 grid(nchunk, 1, batchCount);
    dim3 threads(min(m, MAX_NTHREADS), 1, 1);

    sscal_sger_kernel_batched
        <<< grid, threads, shared_size, queue->cuda_stream() >>>
        (m, n, step, dA_array, lda, info_array, gbstep);
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
        for (i=0; i < n+ib; i++) {
            shared_a[tid + i*ib] = A[tid + i*lda];
        }
    }
    __syncthreads();

    if (tid < n) {
        #pragma unroll
        for (d=0;  d < ib-1; d++) {
            for (i=d+1; i < ib; i++) {
                shared_b[i+tid*ib] += (MAGMA_S_NEG_ONE) * shared_a[i+d*ib] * shared_b[d+tid*ib];
            }
        }
    }
    __syncthreads();

    // write back B
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n; i++) {
            B[tid + i*lda] = shared_b[tid + i*ib];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------

    sgetf2trsm solves one of the matrix equations on gpu

     B = C^-1 * B

    where C, B are part of the matrix A in dA_array,
  
    This version load C, B into shared memory and solve it 
    and copy back to GPU device memory.
    This is an internal routine that might have many assumption.

    Arguments
    ---------
    @param[in]
    ib       INTEGER
            The number of rows/columns of each matrix C, and rows of B.  ib >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix B.  n >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sgesv_aux
    ********************************************************************/

extern "C" void
magma_sgetf2trsm_batched(magma_int_t ib, magma_int_t n, float **dA_array, 
                         magma_int_t step, magma_int_t ldda,
                         magma_int_t batchCount, magma_queue_t queue)
{
    /*
    
    */
    if ( n == 0 || ib == 0 ) return;
    size_t shared_size = sizeof(float)*(ib*(ib+n));

    // TODO TODO TODO
    if ( shared_size > (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 46K leaving 2K for extra
    {
        fprintf( stderr, "%s: error out of shared memory\n", __func__ );
        return;
    }

    dim3 grid(1, 1, batchCount);
    dim3 threads(max(n,ib), 1, 1);

    sgetf2trsm_kernel_batched
        <<< grid, threads, shared_size, queue->cuda_stream() >>>
        (ib, n, dA_array, step, ldda);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ void 
zupdate_device(int m, int step, float* x, int ldx,  float *A, int lda)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );    
    int indx;
    //float reg = MAGMA_S_ZERO;

    // update the current column by all the previous one
    #pragma unroll
    for (int i=0; i < step; i++) {
        for (int s=0; s < nchunk; s++)
        {
            indx = tid + s * MAX_NTHREADS;
            if ( indx > i  && indx < m ) {
                A[indx] -=  A[i] * x[indx + i*ldx];
                //printf("         @ step %d tid %d updating x[tid]*y[i]=A %5.3f %5.3f = %5.3f  at i %d\n", step, tid, x[tid + i*ldx], A[i], A[tid],i);
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
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );    

    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) {
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
    if (info_array[blockIdx.z] != 0 ) return;


    int nchunk = magma_ceildiv( m, MAX_NTHREADS );    
    // read the current column from dev to shared memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) shared_A[tid + s * MAX_NTHREADS] = A0j[tid + s * MAX_NTHREADS];
    }
    __syncthreads();

    // update this column
    if ( step > 0 ) {
        zupdate_device( m, step, A00, lda, shared_A, 1);
        __syncthreads();
    }

    // if ( tid < (m-step) ) // DO NO TPUT THE IF CONDITION HERE SINCE isamax_devfunc HAS __syncthreads INSIDE. 
    // So let all htreads call this routine it will handle correctly based on the size
    // note that isamax need only 128 threads, s
    isamax_devfunc(m-step, shared_A+step, 1, shared_x, shared_idx);
    if (tid == 0) {
        ipiv[gboff]  = shared_idx[0] + gboff + 1; // Fortran Indexing
        alpha = shared_A[shared_idx[0]+step];
        //printf("@ step %d ipiv=%d where gboff=%d  shared_idx %d alpha %5.3f\n",step,ipiv[gboff],gboff,shared_idx[0],alpha);
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[blockIdx.z] = shared_idx[0] + gboff + gbstep + 1;
        }
    }
    __syncthreads();
    if (shared_x[0] == MAGMA_D_ZERO) return;
    __syncthreads();

    // DO NO PUT THE IF CONDITION HERE SINCE isamax_devfunc HAS __syncthreads INSIDE.
    sscal5_device( m-step, shared_A+step, alpha);

    // put back the pivot that has been scaled with itself menaing =1 
    if (tid == 0)  shared_A[shared_idx[0] + step] = alpha;
    __syncthreads();

    // write back from shared to dev memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m )
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
    if ( m == 0) return 0;

    size_t all_shmem_size = zamax*(sizeof(float)+sizeof(int)) + (m+2)*sizeof(float);
    if ( all_shmem_size >  (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 44K leaving 4K for extra
    {
        fprintf( stderr, "%s error out of shared memory\n", __func__ );
        return -20;
    }

    size_t shared_size = sizeof(float)*m;
    dim3 grid(1, 1, batchCount);
    dim3 threads(min(m, MAX_NTHREADS), 1, 1);

    zcomputecolumn_kernel_shared_batched
        <<< grid, threads, shared_size, queue->cuda_stream() >>>
        (m, paneloffset, step, dA_array, lda, ipiv_array, info_array, gbstep);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
kernel_sgetf2_sm_batched(
    magma_int_t m, magma_int_t ib,
    float **dA_array, magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t *info_array)
{
    magma_int_t *ipiv = ipiv_array[blockIdx.z];

    int tx = threadIdx.x;

    float *shared_A = shared_data;

    float *A = dA_array[blockIdx.z];

    float *shared_x = (float*)(shared_A + m * ib);
    int *shared_idx = (int*)(shared_x + zamax);

    float res;

    int length;

    __shared__ int jp;

    // load data to shared memory
    if (tx < m)
    {
        #pragma unroll 8
        for (int i=0; i < ib; i++)
        {
            shared_A[tx + i * m] = A[tx + i * lda];
            //printf("shared_A=%f ", shared_A[tx + i * m]);
        }
    }
    __syncthreads();


    for (int j=0; j < ib; j++)
    {
        length = m - j;

        int offset =  j + j*m;

        //======================================
        //find max
        if (tx < zamax)
        {
            if ( tx < length)
            {
                res = shared_A[tx + offset];
                // printf("res=%f\n", res);
                shared_x[tx] = fabs(MAGMA_S_REAL(res)) + fabs(MAGMA_S_IMAG(res));
                shared_idx[tx] = tx;
            }
            else
            {
                shared_x[tx] = 0.0;
                shared_idx[tx] = 0;
            }
        }
        __syncthreads();


        if (length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
            magma_getidmax<zamax>(tx, shared_x, shared_idx);
        else
            magma_getidmax_n(min(zamax,length), tx, shared_x, shared_idx);

        if (tx == 0)
        {
            jp = shared_idx[0];
           
           
            if (shared_A[jp + offset] == 0.0) printf("error, A(jp,j) == 0.0\n");
           
            ipiv[j]  = j + (jp + 1); // Fortran Indexing 
            //if (blockIdx.x == 1) printf("jp=%d   ", jp + j + 1);
        }
        __syncthreads();
        
        //======================================
        if ( jp != 0) //swap
        {
            if (tx < ib) {
                //printf("A[jp]= %f, A[j]=%f, jp=%d\n", shared_A[jp + j + tx*m], shared_A[j + tx*m], jp);

                float tmp = shared_A[jp + j + tx*m];
                shared_A[jp + j + tx*m] = shared_A[j + tx*m];
                shared_A[j + tx*m] = tmp;
            }
        }
        __syncthreads();

        //======================================
        // Ger
        if (tx < length && tx > 0)
        {
            res = shared_A[tx + offset];
           
            res *= MAGMA_S_DIV(MAGMA_S_ONE, shared_A[0 + offset]); // scaling
           
            shared_A[tx + offset] = res;
           
            #pragma unroll 8
            for (int i=1; i < ib-j; i++) 
            {
                shared_A[tx + i*m + offset] += (MAGMA_S_NEG_ONE) * shared_A[i*m + offset] * res;
           
                //printf("res= %f, shared_A=%f\n", res, shared_A[i*m + offset]);
            }
        }
        __syncthreads();
    } // end of j

    //======================================
    // write back
    if (tx < m)
    {
        #pragma unroll 8
        for (int i=0; i < ib; i++)
        {
            A[tx + i * lda] =  shared_A[tx + i * m];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------
    SGETF2_SM computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    This version load entire matrix (m*ib) into shared memory and factorize it 
    with pivoting and copy back to GPU device memory.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    ib       INTEGER
            The number of columns of each matrix A.  ib >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sgesv_aux
    ********************************************************************/

extern "C"
magma_int_t  magma_sgetf2_sm_batched(
    magma_int_t m, magma_int_t ib,
    float **dA_array, magma_int_t ldda,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, 
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    load entire matrix (m*ib) into shared memory and factorize it with pivoting and copy back.
    */
    size_t shared_size = sizeof(float) * m * ib + (zamax) * (sizeof(float) + sizeof(int)) + sizeof(int);

    if (shared_size > 47000)
    {
        fprintf( stderr, "%s: shared memory = %d, exceeds 48K, kernel cannot run\n", __func__, int(shared_size) );
        return 1;
    }

    dim3 grid(1,1, batchCount);
    dim3 threads(max(max(zamax, m), ib), 1, 1);

    kernel_sgetf2_sm_batched<<< grid, threads, shared_size, queue->cuda_stream() >>>
        ( m, ib, dA_array, ldda, ipiv_array, info_array);

    return 0;
}
