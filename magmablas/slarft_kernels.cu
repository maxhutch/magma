/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from zlarft_kernels.cu normal z -> s, Fri Jan 30 19:00:10 2015
       @author Azzam Haidar
*/

#include "common_magma.h"
#include "magma_templates.h"
#define sgemv_bs 32
#define BLOCK_SIZE 512

#define use_gemm_larft

extern __shared__ float shared_data[];


//===================================================================================================
static __device__
void slarft_gemvcolwise_device( int m, float *v, float *tau,
                         float *c, int ldc, float *T, int ldt, int step )
{

    const int thblk =  blockIdx.x;
    if (thblk > step)
        return;
    /* if blockIdx.x<step step performs the z = V(tx:n,tx)' * V(tx:n,1:tx-1) used for computing T:*/

    if ( !MAGMA_S_EQUAL(*tau, MAGMA_S_ZERO) ) {
        if(thblk<step){    
            const int tx = threadIdx.x;
            float *dc = c + blockIdx.x * ldc;
           
            __shared__ float sum[ BLOCK_SIZE ];
            float tmp;
           
            /* perform  {T_i}^H := V(:,i)' * V(:,1:i-1)  */
            if (tx==0)
                tmp = dc[0]; //since V[0] should be one
            else
                tmp = MAGMA_S_ZERO;
            for( int j = tx+1; j < m; j += BLOCK_SIZE ){
                tmp +=  MAGMA_S_CNJG( v[j] ) * dc[j];
            }
            sum[tx] = tmp;
            magma_sum_reduce< BLOCK_SIZE >( tx, sum );
            #if defined (use_gemm_larft)
            *(T+thblk) = MAGMA_S_CNJG(sum[0]);
            #else
            tmp = - MAGMA_S_CNJG(*tau) * sum[0]; 
            *(T+thblk) = MAGMA_S_CNJG(tmp); // T = - tau(tx) * V(tx:n,1:tx-1)' * V(tx:n,tx) = tmp'
            //*(T+thblk) = - MAGMA_S_CNJG(sum[0]) * (*tau); // T = - tau(tx) * V(tx:n,1:tx-1)' * V(tx:n,tx) = tmp'
            #endif
        }
        else{
            #if defined (use_gemm_larft)
            *(T+thblk) = MAGMA_S_ONE;
            #else
            *(T+thblk) = *tau;
            #endif
        }
    }// in case tau is zero put the corresponding column of T to zero
    else 
    {
        *(T+thblk) = MAGMA_S_ZERO;
    }
}
//===================================================================================================
__global__
void slarft_gemvcolwise_kernel( int m, float *v, int ldv, float *tau,
                          float *T, int ldt, int step )
{
    slarft_gemvcolwise_device(m, v+step+step*ldv, tau+step, v+step, ldv, T+step*ldt, ldt, step);
}
//===================================================================================================
__global__
void slarft_gemvcolwise_kernel_batched( int m, float **v_array, int ldv, float **tau_array,
                          float **T_array, int ldt, int step )
{
    int batchid = blockIdx.z;
    slarft_gemvcolwise_device(m, v_array[batchid]+step+step*ldv, tau_array[batchid]+step, v_array[batchid]+step, ldv, T_array[batchid]+step*ldt, ldt, step);
}
//===================================================================================================
extern "C" 
void magmablas_slarft_gemvcolwise(
    magma_int_t m,  magma_int_t step,
    float *v, magma_int_t ldv, 
    float *T,  magma_int_t ldt,
    float *tau)
{
    dim3 grid( step+1, 1, 1 );
    dim3 threads( BLOCK_SIZE );
    slarft_gemvcolwise_kernel<<< grid, threads, 0, magma_stream >>>( m, v, ldv, tau, T, ldt, step);

}
//===================================================================================================
extern "C" 
void magmablas_slarft_gemvcolwise_batched(
    magma_int_t m,  magma_int_t step,
    float **v_array, magma_int_t ldv, 
    float **T_array,  magma_int_t ldt,
    float **tau_array, magma_int_t batchCount, magma_queue_t queue )
{
    dim3 grid( step+1, 1, batchCount );
    dim3 threads( BLOCK_SIZE );
    slarft_gemvcolwise_kernel_batched<<< grid, threads, 0, queue >>>( m, v_array, ldv, tau_array, T_array, ldt, step);

}
//===================================================================================================




//===================================================================================================
// sgemv(y=alpha*A*x) interface: T/W=tau*v*x, 
static __device__ void
slarft_gemvrowwise_device(
    int m, int i,
    float *tau, 
    float *v_ptr, int ldv, 
    float *x_ptr, int incx,
    float *T_ptr, int ldt,
    float *W, float* sdata)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 


    if(tx ==0 && ty == 0)
    {
        T_ptr[0] = *tau;
    } 

    if(i <= 0) return;
    
    float res = MAGMA_S_ZERO;

    v_ptr += ldv * ty;
            

   
    if(tx < sgemv_bs)
    {
        for(int s=tx; s<m; s+= sgemv_bs)
        {
            res += MAGMA_S_CNJG (v_ptr[s]) * x_ptr[s*incx];
        }
    
        sdata[ty * sgemv_bs + tx] = res;
    }
    __syncthreads();

    magma_sum_reduce<sgemv_bs>(tx, &(sdata[ty*sgemv_bs+0]));

    #if defined (use_gemm_larft)
    if(tx == 0)
    {
            W[ty] = -sdata[ty * sgemv_bs + 0];
    } 
    #else
    if(tx == 0)
    {
            W[ty] = -sdata[ty * sgemv_bs + 0] * (*tau) ;
    }
    #endif 
}




//T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
//T(i,i) = tau(i)
//===================================================================================================
 __global__ void
slarft_gemvrowwise_kernel(
    int m, int i, 
    float *tau, 
    float *v, int ldv, 
    float *T, int ldt)
{

    float *W =  T +i*ldt;

    float *sdata = (float*)shared_data;

    slarft_gemvrowwise_device(m, i, tau+i, v+i, ldv,  v+i+i*ldv, 1,  
                           T+i+i*ldt , ldt, W, sdata);
}

//===================================================================================================
__global__ void
slarft_gemvrowwise_kernel_batched(
    int m, int i,
    float **tau_array, 
    float **v_array, int ldv, 
    float **T_array, int ldt)
{

    int batchid = blockIdx.z;

    float *W =  T_array[batchid] +i*ldt;

    float *sdata = (float*)shared_data;

    slarft_gemvrowwise_device(m, i, tau_array[batchid]+i, v_array[batchid]+i, ldv,  v_array[batchid]+i+i*ldv, 1,  
                           T_array[batchid] +i+i*ldt , ldt, W, sdata);
}

//===================================================================================================
extern "C"
void magmablas_slarft_gemvrowwise(
    magma_int_t m, magma_int_t i, 
    float *tau, 
    float *v, magma_int_t ldv, 
    float *T, magma_int_t ldt,
    float *W)
{

    dim3 grid(1);


    dim3 threads(sgemv_bs, max(i,1), 1);


    slarft_gemvrowwise_kernel <<< grid, threads, sizeof(float)*sgemv_bs*(i+1), magma_stream>>>(m, i, tau, v, ldv, T, ldt);
}
//===================================================================================================
extern "C"
void magmablas_slarft_gemvrowwise_batched(
    magma_int_t m, magma_int_t i, 
    float **tau_array, 
    float **v_array, magma_int_t ldv, 
    float **T_array, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(sgemv_bs, max(i,1), 1);

    /*  sgemvrowwise used a bigger shared memory and has more data reuse and performs better
    */
    slarft_gemvrowwise_kernel_batched <<< grid, threads, sizeof(float)*sgemv_bs*(i+1), queue>>>(m, i,  tau_array, v_array, ldv, T_array, ldt);
}
//===================================================================================================
   


//===================================================================================================
/*
   loop_inside
*/
static __device__ void
slarft_gemv_loop_inside_device(
    int n, int k, 
    float *tau, 
    float *v, int ldv, 
    float *T, int ldt)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    
    int incx = 1;
    float *sdata = (float*)shared_data;

    float res;

    // write the first elment
    if(tx ==0 && ty == 0)
    {
        T[0] = tau[0];
    } 
 
    for(int i=1; i<k;i++)
    {

        int m = n-i; 

        float *v_ptr = v;

        v_ptr += i;

        float *x_ptr = v_ptr + i * ldv;
            
        res = MAGMA_S_ZERO;
            
        if(tx < sgemv_bs && ty < i)
        {
            v_ptr += ldv * ty;

            for(int s=tx; s<m; s+= sgemv_bs)
            {
                res += MAGMA_S_CNJG (v_ptr[s]) * x_ptr[s*incx];
            }
    
            sdata[ty * sgemv_bs + tx] = res;
        }
        __syncthreads();

        magma_sum_reduce<sgemv_bs>(tx, &(sdata[ty*sgemv_bs+0]));
        

       __syncthreads();
       #if defined (use_gemm_larft)
       if(tx < i && ty == 0)
       {
            T[i* ldt + tx] = sdata[tx * sgemv_bs + 0];  
       } 
       // not needed since it is overwritten in trmv
       /*
       if(tx == i && ty == 0)
       {
           T[i * ldt + i] = tau[i];
       }
       */
       #else
       if(tx < i && ty == 0)
       {
           T[i* ldt + tx] = -sdata[tx * sgemv_bs + 0] * (tau[i]) ;  
       } 
      
       if(tx == i && ty == 0)
       {
           T[i * ldt + i] = tau[i];
       }
       #endif
     
       v_ptr -= i;

    }// end of loop k
}
//===================================================================================================
__global__ void
slarft_gemv_loop_inside_kernel(
    int n, int k, 
    float *tau, 
    float *v, int ldv, 
    float *T, int ldt)
{
    slarft_gemv_loop_inside_device(n, k, tau, v, ldv, T, ldt);
}
//===================================================================================================
__global__ void
slarft_gemv_loop_inside_kernel_batched(
    int n, int k, 
    float **tau_array, 
    float **v_array, int ldv, 
    float **T_array, int ldt)
{
    int batchid = blockIdx.z;
    slarft_gemv_loop_inside_device(n, k, tau_array[batchid], v_array[batchid], ldv, T_array[batchid], ldt);
}
//===================================================================================================
//===================================================================================================
//===================================================================================================
extern "C"
void magmablas_slarft_gemv_loop_inside(
    int n, int k, 
    float *tau, 
    float *v, int ldv, 
    float *T, int ldt)
{

    dim3 grid(1);
    dim3 threads(sgemv_bs, max(k,1), 1);
    slarft_gemv_loop_inside_kernel<<<grid, threads, sizeof(float) * (sgemv_bs*(k+1)), magma_stream>>>(n, k, tau, v, ldv, T, ldt); 
}
//===================================================================================================
extern "C"
void magmablas_slarft_gemv_loop_inside_batched(
    int n, int k, 
    float **tau_array, 
    float **v_array, int ldv, 
    float **T_array, int ldt, magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(sgemv_bs, max(k,1), 1);
    slarft_gemv_loop_inside_kernel_batched<<<grid, threads, sizeof(float) * (sgemv_bs*(k+1)), queue>>>(n, k, tau_array, v_array, ldv, T_array, ldt); 
}
//===================================================================================================





//===================================================================================================
static  __device__ void 
slarft_strmv_sm32x32_device(
    int n, int k, float *tau,
    float *Tin, int ldtin,  float *Tout, int ldtout )
{
    int tx = threadIdx.x; 
    float *sdata = (float*)shared_data;
    float res;

    // this routine apply a sequence of trmv to update k column of the triangular
    // T starting at n-k to n where T is of size n by n and where the first n-k 
    // columns of T are supposed updated previously.
    // So the routine load all of T nxn to the shared memory 
    // and apply the sequence of trmv.
    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate 
    // one element of the column of T then move to the next column

    // read T into shared
    for(int s=0; s<n-k; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }
    
#if defined(use_gemm_larft)
    for(int s=n-k; s<n; s++)
    {
        if(tx == s)
            sdata[tx + s*n] = tau[s];
        else
            sdata[tx + s*n] = -tau[s] * Tin[tx + s * ldtin];
    }
#else
    for(int s=n-k; s<n; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }
#endif

    // perform trmv
    for(int i=n-k; i<n;i++)
    {
       __syncthreads();  
       res = MAGMA_S_ZERO;
       if(tx < i)
       {
           for(int j=tx; j<i; j++)
           {
               res += sdata[tx + j * n] * sdata[j+ i * n];      
           }
       }       
       __syncthreads();  
       if(tx < i)
       {
           sdata[tx + i * n] = res;
       }
    } 

    __syncthreads();  
    // write back the updated block of k column of T
    for(int s=n-k; s<n; s++)
    {
       Tout[tx + s * ldtout] = sdata[tx + s*n];
    }

}
//===================================================================================================
__global__ void 
slarft_strmv_sm32x32_kernel(
    int n, int k, float *tau,
    float *Tin, int ldtin,  float *Tout, int ldtout )
{
    slarft_strmv_sm32x32_device( n, k, tau, Tin, ldtin, Tout, ldtout);
}
//===================================================================================================
__global__ void 
slarft_strmv_sm32x32_kernel_batched(
    int n, int k, float **tau_array,
    float **Tin_array, int ldtin,  float **Tout_array, int ldtout )
{
    int batchId = blockIdx.z;
    slarft_strmv_sm32x32_device( n, k, tau_array[batchId], Tin_array[batchId], ldtin, Tout_array[batchId], ldtout);
}
//===================================================================================================
//===================================================================================================
extern "C"
void magmablas_slarft_strmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    float *tau, 
    float *Tin, magma_int_t ldtin, 
    float *Tout, magma_int_t ldtout)
{

    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    slarft_strmv_sm32x32_kernel <<< grid, threads, sizeof(float)*(m*m), magma_stream >>> (m, n,  tau, Tin, ldtin, Tout, ldtout);
}
//===================================================================================================
extern "C"
void magmablas_slarft_strmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    float **tau_array, 
    float **Tin_array, magma_int_t ldtin, 
    float **Tout_array, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    slarft_strmv_sm32x32_kernel_batched <<< grid, threads, sizeof(float)*(m*m), queue >>> (m, n,  tau_array, Tin_array, ldtin, Tout_array, ldtout);
}
//===================================================================================================




//===================================================================================================
//===================================================================================================
static __device__ void 
slarft_recstrmv_sm32x32_device(
    int m, int n, float *tau,
    float *Trec, int ldtrec, float *Ttri, int ldttri)
{
    int tx = threadIdx.x; 
    float *sdata = (float*)shared_data;
    float res;

    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate 
    // one element of the column of T then move to the next column

    // read T into shared
    for(int s=0; s<n; s++)
    {
        sdata[tx + s*n] = Trec[tx + s * ldtrec];
    }
    __syncthreads();  
    
    // perform sequence of n-1 gemv
    for(int i=0; i<n;i++)
    {
       res = MAGMA_S_ZERO;
       for(int j=0; j<i; j++)
       {
           res += sdata[tx + j * n] * Ttri[j+ i * ldttri];      
       }
       __syncthreads();   // a enlever
       sdata[tx + i * n] = -tau[i] * (sdata[tx + i * n] + res);
       __syncthreads();  
    } 

    // write back the updated block of k column of T  multiplying by -tau
    for(int s=0; s<n; s++)
    {
       Trec[tx + s * ldtrec] = sdata[tx + s*n];
    }

}

//===================================================================================================
__global__ void 
slarft_recstrmv_sm32x32_kernel(
    int m, int n, float *tau,
    float *Trec, int ldtrec, float *Ttri, int ldttri)
{
    slarft_recstrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri);
}
//===================================================================================================
__global__ void 
slarft_recstrmv_sm32x32_kernel_batched(
    int m, int n, float **tau_array,
    float **Trec_array, int ldtrec, float **Ttri_array, int ldttri)
{
    int batchId = blockIdx.z;
    slarft_recstrmv_sm32x32_device(m, n, tau_array[batchId], Trec_array[batchId], ldtrec, Ttri_array[batchId], ldttri);
}
//===================================================================================================
extern "C"
void magmablas_slarft_recstrmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    float *tau, 
    float *Trec, magma_int_t ldtrec, 
    float *Ttri, magma_int_t ldttri)
{

    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    slarft_recstrmv_sm32x32_kernel <<< grid, threads, sizeof(float)*(m*n), magma_stream >>> (m, n,  tau, Trec, ldtrec, Ttri, ldttri);
}
//===================================================================================================
extern "C"
void magmablas_slarft_recstrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    float **tau_array, 
    float **Trec_array, magma_int_t ldtrec, 
    float **Ttri_array, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    slarft_recstrmv_sm32x32_kernel_batched <<< grid, threads, sizeof(float)*(m*n), queue >>> (m, n,  tau_array, Trec_array, ldtrec, Ttri_array, ldttri);
}
//===================================================================================================


