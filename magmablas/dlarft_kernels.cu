/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from zlarft_kernels.cu normal z -> d, Fri Jan 30 19:00:10 2015
       @author Azzam Haidar
*/

#include "common_magma.h"
#include "magma_templates.h"
#define dgemv_bs 32
#define BLOCK_SIZE 512

#define use_gemm_larft

extern __shared__ double shared_data[];


//===================================================================================================
static __device__
void dlarft_gemvcolwise_device( int m, double *v, double *tau,
                         double *c, int ldc, double *T, int ldt, int step )
{

    const int thblk =  blockIdx.x;
    if (thblk > step)
        return;
    /* if blockIdx.x<step step performs the z = V(tx:n,tx)' * V(tx:n,1:tx-1) used for computing T:*/

    if ( !MAGMA_D_EQUAL(*tau, MAGMA_D_ZERO) ) {
        if(thblk<step){    
            const int tx = threadIdx.x;
            double *dc = c + blockIdx.x * ldc;
           
            __shared__ double sum[ BLOCK_SIZE ];
            double tmp;
           
            /* perform  {T_i}^H := V(:,i)' * V(:,1:i-1)  */
            if (tx==0)
                tmp = dc[0]; //since V[0] should be one
            else
                tmp = MAGMA_D_ZERO;
            for( int j = tx+1; j < m; j += BLOCK_SIZE ){
                tmp +=  MAGMA_D_CNJG( v[j] ) * dc[j];
            }
            sum[tx] = tmp;
            magma_sum_reduce< BLOCK_SIZE >( tx, sum );
            #if defined (use_gemm_larft)
            *(T+thblk) = MAGMA_D_CNJG(sum[0]);
            #else
            tmp = - MAGMA_D_CNJG(*tau) * sum[0]; 
            *(T+thblk) = MAGMA_D_CNJG(tmp); // T = - tau(tx) * V(tx:n,1:tx-1)' * V(tx:n,tx) = tmp'
            //*(T+thblk) = - MAGMA_D_CNJG(sum[0]) * (*tau); // T = - tau(tx) * V(tx:n,1:tx-1)' * V(tx:n,tx) = tmp'
            #endif
        }
        else{
            #if defined (use_gemm_larft)
            *(T+thblk) = MAGMA_D_ONE;
            #else
            *(T+thblk) = *tau;
            #endif
        }
    }// in case tau is zero put the corresponding column of T to zero
    else 
    {
        *(T+thblk) = MAGMA_D_ZERO;
    }
}
//===================================================================================================
__global__
void dlarft_gemvcolwise_kernel( int m, double *v, int ldv, double *tau,
                          double *T, int ldt, int step )
{
    dlarft_gemvcolwise_device(m, v+step+step*ldv, tau+step, v+step, ldv, T+step*ldt, ldt, step);
}
//===================================================================================================
__global__
void dlarft_gemvcolwise_kernel_batched( int m, double **v_array, int ldv, double **tau_array,
                          double **T_array, int ldt, int step )
{
    int batchid = blockIdx.z;
    dlarft_gemvcolwise_device(m, v_array[batchid]+step+step*ldv, tau_array[batchid]+step, v_array[batchid]+step, ldv, T_array[batchid]+step*ldt, ldt, step);
}
//===================================================================================================
extern "C" 
void magmablas_dlarft_gemvcolwise(
    magma_int_t m,  magma_int_t step,
    double *v, magma_int_t ldv, 
    double *T,  magma_int_t ldt,
    double *tau)
{
    dim3 grid( step+1, 1, 1 );
    dim3 threads( BLOCK_SIZE );
    dlarft_gemvcolwise_kernel<<< grid, threads, 0, magma_stream >>>( m, v, ldv, tau, T, ldt, step);

}
//===================================================================================================
extern "C" 
void magmablas_dlarft_gemvcolwise_batched(
    magma_int_t m,  magma_int_t step,
    double **v_array, magma_int_t ldv, 
    double **T_array,  magma_int_t ldt,
    double **tau_array, magma_int_t batchCount, magma_queue_t queue )
{
    dim3 grid( step+1, 1, batchCount );
    dim3 threads( BLOCK_SIZE );
    dlarft_gemvcolwise_kernel_batched<<< grid, threads, 0, queue >>>( m, v_array, ldv, tau_array, T_array, ldt, step);

}
//===================================================================================================




//===================================================================================================
// dgemv(y=alpha*A*x) interface: T/W=tau*v*x, 
static __device__ void
dlarft_gemvrowwise_device(
    int m, int i,
    double *tau, 
    double *v_ptr, int ldv, 
    double *x_ptr, int incx,
    double *T_ptr, int ldt,
    double *W, double* sdata)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 


    if(tx ==0 && ty == 0)
    {
        T_ptr[0] = *tau;
    } 

    if(i <= 0) return;
    
    double res = MAGMA_D_ZERO;

    v_ptr += ldv * ty;
            

   
    if(tx < dgemv_bs)
    {
        for(int s=tx; s<m; s+= dgemv_bs)
        {
            res += MAGMA_D_CNJG (v_ptr[s]) * x_ptr[s*incx];
        }
    
        sdata[ty * dgemv_bs + tx] = res;
    }
    __syncthreads();

    magma_sum_reduce<dgemv_bs>(tx, &(sdata[ty*dgemv_bs+0]));

    #if defined (use_gemm_larft)
    if(tx == 0)
    {
            W[ty] = -sdata[ty * dgemv_bs + 0];
    } 
    #else
    if(tx == 0)
    {
            W[ty] = -sdata[ty * dgemv_bs + 0] * (*tau) ;
    }
    #endif 
}




//T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
//T(i,i) = tau(i)
//===================================================================================================
 __global__ void
dlarft_gemvrowwise_kernel(
    int m, int i, 
    double *tau, 
    double *v, int ldv, 
    double *T, int ldt)
{

    double *W =  T +i*ldt;

    double *sdata = (double*)shared_data;

    dlarft_gemvrowwise_device(m, i, tau+i, v+i, ldv,  v+i+i*ldv, 1,  
                           T+i+i*ldt , ldt, W, sdata);
}

//===================================================================================================
__global__ void
dlarft_gemvrowwise_kernel_batched(
    int m, int i,
    double **tau_array, 
    double **v_array, int ldv, 
    double **T_array, int ldt)
{

    int batchid = blockIdx.z;

    double *W =  T_array[batchid] +i*ldt;

    double *sdata = (double*)shared_data;

    dlarft_gemvrowwise_device(m, i, tau_array[batchid]+i, v_array[batchid]+i, ldv,  v_array[batchid]+i+i*ldv, 1,  
                           T_array[batchid] +i+i*ldt , ldt, W, sdata);
}

//===================================================================================================
extern "C"
void magmablas_dlarft_gemvrowwise(
    magma_int_t m, magma_int_t i, 
    double *tau, 
    double *v, magma_int_t ldv, 
    double *T, magma_int_t ldt,
    double *W)
{

    dim3 grid(1);


    dim3 threads(dgemv_bs, max(i,1), 1);


    dlarft_gemvrowwise_kernel <<< grid, threads, sizeof(double)*dgemv_bs*(i+1), magma_stream>>>(m, i, tau, v, ldv, T, ldt);
}
//===================================================================================================
extern "C"
void magmablas_dlarft_gemvrowwise_batched(
    magma_int_t m, magma_int_t i, 
    double **tau_array, 
    double **v_array, magma_int_t ldv, 
    double **T_array, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(dgemv_bs, max(i,1), 1);

    /*  dgemvrowwise used a bigger shared memory and has more data reuse and performs better
    */
    dlarft_gemvrowwise_kernel_batched <<< grid, threads, sizeof(double)*dgemv_bs*(i+1), queue>>>(m, i,  tau_array, v_array, ldv, T_array, ldt);
}
//===================================================================================================
   


//===================================================================================================
/*
   loop_inside
*/
static __device__ void
dlarft_gemv_loop_inside_device(
    int n, int k, 
    double *tau, 
    double *v, int ldv, 
    double *T, int ldt)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    
    int incx = 1;
    double *sdata = (double*)shared_data;

    double res;

    // write the first elment
    if(tx ==0 && ty == 0)
    {
        T[0] = tau[0];
    } 
 
    for(int i=1; i<k;i++)
    {

        int m = n-i; 

        double *v_ptr = v;

        v_ptr += i;

        double *x_ptr = v_ptr + i * ldv;
            
        res = MAGMA_D_ZERO;
            
        if(tx < dgemv_bs && ty < i)
        {
            v_ptr += ldv * ty;

            for(int s=tx; s<m; s+= dgemv_bs)
            {
                res += MAGMA_D_CNJG (v_ptr[s]) * x_ptr[s*incx];
            }
    
            sdata[ty * dgemv_bs + tx] = res;
        }
        __syncthreads();

        magma_sum_reduce<dgemv_bs>(tx, &(sdata[ty*dgemv_bs+0]));
        

       __syncthreads();
       #if defined (use_gemm_larft)
       if(tx < i && ty == 0)
       {
            T[i* ldt + tx] = sdata[tx * dgemv_bs + 0];  
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
           T[i* ldt + tx] = -sdata[tx * dgemv_bs + 0] * (tau[i]) ;  
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
dlarft_gemv_loop_inside_kernel(
    int n, int k, 
    double *tau, 
    double *v, int ldv, 
    double *T, int ldt)
{
    dlarft_gemv_loop_inside_device(n, k, tau, v, ldv, T, ldt);
}
//===================================================================================================
__global__ void
dlarft_gemv_loop_inside_kernel_batched(
    int n, int k, 
    double **tau_array, 
    double **v_array, int ldv, 
    double **T_array, int ldt)
{
    int batchid = blockIdx.z;
    dlarft_gemv_loop_inside_device(n, k, tau_array[batchid], v_array[batchid], ldv, T_array[batchid], ldt);
}
//===================================================================================================
//===================================================================================================
//===================================================================================================
extern "C"
void magmablas_dlarft_gemv_loop_inside(
    int n, int k, 
    double *tau, 
    double *v, int ldv, 
    double *T, int ldt)
{

    dim3 grid(1);
    dim3 threads(dgemv_bs, max(k,1), 1);
    dlarft_gemv_loop_inside_kernel<<<grid, threads, sizeof(double) * (dgemv_bs*(k+1)), magma_stream>>>(n, k, tau, v, ldv, T, ldt); 
}
//===================================================================================================
extern "C"
void magmablas_dlarft_gemv_loop_inside_batched(
    int n, int k, 
    double **tau_array, 
    double **v_array, int ldv, 
    double **T_array, int ldt, magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(dgemv_bs, max(k,1), 1);
    dlarft_gemv_loop_inside_kernel_batched<<<grid, threads, sizeof(double) * (dgemv_bs*(k+1)), queue>>>(n, k, tau_array, v_array, ldv, T_array, ldt); 
}
//===================================================================================================





//===================================================================================================
static  __device__ void 
dlarft_dtrmv_sm32x32_device(
    int n, int k, double *tau,
    double *Tin, int ldtin,  double *Tout, int ldtout )
{
    int tx = threadIdx.x; 
    double *sdata = (double*)shared_data;
    double res;

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
       res = MAGMA_D_ZERO;
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
dlarft_dtrmv_sm32x32_kernel(
    int n, int k, double *tau,
    double *Tin, int ldtin,  double *Tout, int ldtout )
{
    dlarft_dtrmv_sm32x32_device( n, k, tau, Tin, ldtin, Tout, ldtout);
}
//===================================================================================================
__global__ void 
dlarft_dtrmv_sm32x32_kernel_batched(
    int n, int k, double **tau_array,
    double **Tin_array, int ldtin,  double **Tout_array, int ldtout )
{
    int batchId = blockIdx.z;
    dlarft_dtrmv_sm32x32_device( n, k, tau_array[batchId], Tin_array[batchId], ldtin, Tout_array[batchId], ldtout);
}
//===================================================================================================
//===================================================================================================
extern "C"
void magmablas_dlarft_dtrmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    double *tau, 
    double *Tin, magma_int_t ldtin, 
    double *Tout, magma_int_t ldtout)
{

    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    dlarft_dtrmv_sm32x32_kernel <<< grid, threads, sizeof(double)*(m*m), magma_stream >>> (m, n,  tau, Tin, ldtin, Tout, ldtout);
}
//===================================================================================================
extern "C"
void magmablas_dlarft_dtrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    double **tau_array, 
    double **Tin_array, magma_int_t ldtin, 
    double **Tout_array, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    dlarft_dtrmv_sm32x32_kernel_batched <<< grid, threads, sizeof(double)*(m*m), queue >>> (m, n,  tau_array, Tin_array, ldtin, Tout_array, ldtout);
}
//===================================================================================================




//===================================================================================================
//===================================================================================================
static __device__ void 
dlarft_recdtrmv_sm32x32_device(
    int m, int n, double *tau,
    double *Trec, int ldtrec, double *Ttri, int ldttri)
{
    int tx = threadIdx.x; 
    double *sdata = (double*)shared_data;
    double res;

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
       res = MAGMA_D_ZERO;
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
dlarft_recdtrmv_sm32x32_kernel(
    int m, int n, double *tau,
    double *Trec, int ldtrec, double *Ttri, int ldttri)
{
    dlarft_recdtrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri);
}
//===================================================================================================
__global__ void 
dlarft_recdtrmv_sm32x32_kernel_batched(
    int m, int n, double **tau_array,
    double **Trec_array, int ldtrec, double **Ttri_array, int ldttri)
{
    int batchId = blockIdx.z;
    dlarft_recdtrmv_sm32x32_device(m, n, tau_array[batchId], Trec_array[batchId], ldtrec, Ttri_array[batchId], ldttri);
}
//===================================================================================================
extern "C"
void magmablas_dlarft_recdtrmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    double *tau, 
    double *Trec, magma_int_t ldtrec, 
    double *Ttri, magma_int_t ldttri)
{

    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    dlarft_recdtrmv_sm32x32_kernel <<< grid, threads, sizeof(double)*(m*n), magma_stream >>> (m, n,  tau, Trec, ldtrec, Ttri, ldttri);
}
//===================================================================================================
extern "C"
void magmablas_dlarft_recdtrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    double **tau_array, 
    double **Trec_array, magma_int_t ldtrec, 
    double **Ttri_array, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue)
{

    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    dlarft_recdtrmv_sm32x32_kernel_batched <<< grid, threads, sizeof(double)*(m*n), queue >>> (m, n,  tau_array, Trec_array, ldtrec, Ttri_array, ldttri);
}
//===================================================================================================


