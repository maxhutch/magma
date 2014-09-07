/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define PRECISION_z

#define hemv_bs          32
#define bank_shift       33

/*******************************************************************************
 *    Lower case, where n is multiple of block size (hemv_bs)
 */

__global__ void
zhemv_kernel_fermi_L_special_mgpu_offset_32(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex  beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan)
{
#if (__CUDA_ARCH__ >= 200)
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    if ( blkc < my_gpu_id ) {
        return;
    }

    magmaDoubleComplex res  = MAGMA_Z_ZERO; // used in scan the row
    magmaDoubleComplex res_ = MAGMA_Z_ZERO; // used in scan the column
    magmaDoubleComplex res1 = MAGMA_Z_ZERO; // tem for res
    magmaDoubleComplex res2 = MAGMA_Z_ZERO; // tem for res_

    __shared__ magmaDoubleComplex la   [hemv_bs][bank_shift];
    __shared__ magmaDoubleComplex sdata   [hemv_bs][9];
    __shared__ magmaDoubleComplex buff [hemv_bs];
    __shared__ magmaDoubleComplex buff2 [hemv_bs];

    int break_d   =  hemv_bs * blkc;

    x  += (break_d + tx) * incx;
    A  +=  break_d;
    A  +=  ty * lda + tx;

    if ( ty == 0 ) {
        buff[tx] = x[0];
        if ( blkc == 0 && my_gpu_id == 0 && tx < kstan ) {
            buff[tx] = MAGMA_Z_ZERO;
        }
    } // obtain the vector x store in buff;

    int flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id ) {
        A += lda * (blkc/num_gpus) * hemv_bs; // change

        #pragma unroll
        for(int j=0; j < hemv_bs; j += 8)
            la[0][ bank_shift * (ty+j) + tx] = A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(int i=ty*4; i < (ty*4 + 4); i++) {
            if ( i < tx ) {
                la[0][bank_shift * tx + i] = cuConj( la[0][ i * bank_shift + tx] );
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj( la[0][bank_shift * tx + j + ty*4] ) * buff[j + ty*4];
        __syncthreads();

        A -= lda * (blkc/num_gpus) * hemv_bs;
            
        flag = 1;
    }

    x -= blkc * hemv_bs * incx;

    x = x - tx*incx;

    int wc_c = my_gpu_id;
    int count = 0;

    WC +=  break_d + tx;
    
    int num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if ( my_gpu_id < ( (blkc+1) % num_gpus) ) {
        num_blocks_iters += 1;
    }

    x += (my_gpu_id) * hemv_bs;

    if ( blkc > my_gpu_id ) {
        for(int s=0; s < num_blocks_iters; s++) {
            res_ = MAGMA_Z_ZERO;
            count++;

            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8)
                la[0][ bank_shift * (ty+j) + tx] = A[ j * lda];

            if ( ty == 0 ) {
                buff2[tx] = x[tx];
                if ( my_gpu_id == 0 && tx < kstan && count == 1 ) {
                    buff2[tx] = MAGMA_Z_ZERO;
                }
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(int j=0; j < 4; j++) {
                res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                res_ += cuConj( la[0][bank_shift * tx + j + ty*4] ) * buff[j + ty*4]; //iterate colum
            }

            sdata[tx][ty] = res_;
            __syncthreads();

            if ( ty == 1 ) {
                res2 = sdata[tx][0]+sdata[tx][1]
                     + sdata[tx][2]+sdata[tx][3]
                     + sdata[tx][4]+sdata[tx][5]
                     + sdata[tx][6]+sdata[tx][7];
                
                WC[wc_c*lda ] =   res2;
            }

            wc_c += num_gpus;
            x += num_gpus * hemv_bs;
            A += lda * hemv_bs;
        }
    }

    la[0][bank_shift*tx+ty] = res;
    __syncthreads();
    
    if ( ty == 0 ) {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
        
        WC[0+lda*(blkc)] =  res1;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

/**************************************************************
 *    Lower case for generic sizes
 */
__global__ void
zhemv_kernel_fermi_L_generic_mgpu_offset_32(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int m_mod_nb,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan)
{
#if (__CUDA_ARCH__ >= 200)
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    if ( blkc < my_gpu_id ) {
        return;
    }

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [hemv_bs][bank_shift];
    __shared__ magmaDoubleComplex sdata   [hemv_bs][9];
    __shared__ magmaDoubleComplex buff [hemv_bs];
    __shared__ magmaDoubleComplex buff2 [hemv_bs];

    int break_d   =  hemv_bs * blkc;

    x += (break_d + tx) * incx;
    A +=  break_d;
    A += lda * ty;

    int trackA;
    if ( blkc == ( gridDim.x - 1 ) ) {
        if ( ty == 0 ) {
            if ( tx > m_mod_nb ) {
                buff[tx] = MAGMA_Z_ZERO;
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_nb )
            trackA=m_mod_nb;
        else
            trackA=tx;
        A += trackA;
    }
    else {
        if ( ty == 0 ) {
            buff[tx]  = x[0];
        }
        trackA = tx;
        A += trackA;
    }

    if ( ty == 0 ) {
        if ( my_gpu_id == 0 && blkc == 0  && tx < kstan ) {
            buff[tx] = MAGMA_Z_ZERO;
        }
    }

    int flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id ) {
        A += lda * (blkc/num_gpus) * hemv_bs; // change
        // Somehow merging these two if - else creates problem
        // It could be a potential bug -- from synchronization or from cuda or compiler
        if ( blkc == ( gridDim.x - 1 ) ) {
            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8) {
                if ( ( ty + j ) > m_mod_nb ) {
                    la[0][bank_shift*(ty+j)+tx] = MAGMA_Z_MAKE( 9999, 0 );
                }
                else
                    la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
        }
        else {
            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8) {
                la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
        }
        __syncthreads();

        #pragma unroll
        for(int i=ty*4; i < (ty*4+4); i++) {
            if ( i < tx ) {
                la[0][bank_shift*tx+i] = cuConj(la[0][i*bank_shift+tx]);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj(la[0][bank_shift*tx+j+ty*4])* buff[j+ty*4];
        __syncthreads();

        A -= lda * (blkc/num_gpus) * hemv_bs;
        
        flag = 1;
    }

    __syncthreads();

    x = x - break_d*incx;
    x = x - tx * incx;

    int wc_c = my_gpu_id;
    int count = 0;

    WC +=  break_d + tx;

    int num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if ( my_gpu_id < ( (blkc+1) % num_gpus) ) {
        num_blocks_iters += 1;
    }

    x += (my_gpu_id) * hemv_bs;

    if ( blkc > my_gpu_id ) {
        for(int s=0; s < num_blocks_iters; s++) {
            res_ = MAGMA_Z_ZERO;
            count++;

            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8)
                la[0][ bank_shift * (ty+j) + tx] = A[ j * lda];

            if ( ty == 0 ) {
                buff2[tx] = x[tx];
                if ( my_gpu_id == 0 && tx < kstan && count == 1 ) {
                    buff2[tx] = MAGMA_Z_ZERO;
                }
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(int j=0; j < 4; j++) {
                res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                res_ += cuConj( la[0][bank_shift * tx + j + ty*4] ) * buff[j + ty*4]; //iterate colum
            }

            sdata[tx][ty] = res_;
            __syncthreads();

            if ( ty == 1 ) {
                res2 = sdata[tx][0]+sdata[tx][1]
                     + sdata[tx][2]+sdata[tx][3]
                     + sdata[tx][4]+sdata[tx][5]
                     + sdata[tx][6]+sdata[tx][7];
                
                WC[wc_c*lda ] =   res2;
            }

            wc_c += num_gpus;
            x += num_gpus * hemv_bs;
            A += lda * hemv_bs;
        }
    }

    la[0][bank_shift*tx+ty] = res;
    __syncthreads();
    
    if ( ty == 0 ) {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
        WC[0+lda*(blkc)] =  res1;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/**************************************************************
 *
 */

__global__ void
zhemv_kernel_fermi_L_update_mgpu_offset_32(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan )
{
#if (__CUDA_ARCH__ >= 200)
    int i;
    int tx  = threadIdx.x;
    int ind = blockIdx.x * hemv_bs + tx;
    magmaDoubleComplex Ca;

    Ca = MAGMA_Z_ZERO;
    WC += ind + lda * blockIdx.x;

    for(i = blockIdx.x * hemv_bs; i < n; i += hemv_bs) {
        Ca += WC[0];
        WC += hemv_bs;
    }
    if ( ind < n && ind >= kstan )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca;
#endif /* (__CUDA_ARCH__ >= 200) */
}


extern "C"
void magmablas_zhemv_fermi_L_mgpu_offset_32(
    magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, magma_int_t incy,
    magmaDoubleComplex *dwork,
    magma_int_t my_gpu_id,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_int_t num_blocks_skipped)
{
    magma_int_t the_chosen_block_id = offset / nb;
    magma_int_t kstan = offset % nb;

    A += lda * num_blocks_skipped * nb + the_chosen_block_id * nb;
    x += the_chosen_block_id * nb;
    y += the_chosen_block_id * nb;

    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    blocks -= the_chosen_block_id;

    dim3 grid(blocks, 1, 1);
    dim3 threads(nb, 8, 1);
    dim3 threads_u(nb, 1, 1);

    /*
     * If matrix size is multiple of hemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if ( n % hemv_bs == 0 ) {
        zhemv_kernel_fermi_L_special_mgpu_offset_32<<< grid, threads, 0, magma_stream >>>(
            n, alpha, A, lda, x, incx, beta, y, incy, dwork,
            my_gpu_id, num_gpus, nb, kstan);
    }
    else {
        magma_int_t m_mod_nb = (n % hemv_bs) - 1;
        zhemv_kernel_fermi_L_generic_mgpu_offset_32<<< grid, threads, 0, magma_stream >>>(
            n, alpha, A, lda, x, incx, beta, y, incy, dwork,
            m_mod_nb, my_gpu_id, num_gpus, nb, kstan);
    }

    zhemv_kernel_fermi_L_update_mgpu_offset_32<<< grid, threads_u, 0, magma_stream >>>(
        n, alpha, A, lda, x, incx, beta, y, incy, dwork,
        my_gpu_id, num_gpus, nb, kstan);
}

/*******************************************************************************
 *    Upper case, where n is multiple of block size (hemv_bs)
 */

__global__ void
zhemv_kernel_fermi_U_special_mgpu_offset_32(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex  beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan)
{
#if (__CUDA_ARCH__ >= 200)
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    magmaDoubleComplex res  = MAGMA_Z_ZERO; // used in scan the row
    magmaDoubleComplex res_ = MAGMA_Z_ZERO; // used in scan the column
    magmaDoubleComplex res1 = MAGMA_Z_ZERO; // tem for res
    magmaDoubleComplex res2 = MAGMA_Z_ZERO; // tem for res_

    __shared__ magmaDoubleComplex la   [hemv_bs][bank_shift];
    __shared__ magmaDoubleComplex buff [hemv_bs];
    __shared__ magmaDoubleComplex buff2 [hemv_bs];

    int break_d   =  hemv_bs * blkc;

    x  += (break_d + tx) * incx;
    A  +=  break_d;
    A  +=  ty * lda + tx;

    if ( ty == 0 ) {
        buff[tx] = x[0];
        if ( blkc == 0  && tx < kstan ) {
            buff[tx] = MAGMA_Z_ZERO;
        }
    } // obtain the vector x store in buff;

    if ( (blkc % num_gpus) == my_gpu_id ) {
        A += lda * (blkc/num_gpus) * hemv_bs; // change

        #pragma unroll
        for(int j=0; j < hemv_bs; j += 8)
            la[0][ bank_shift * (ty+j) + tx] = A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(int i=ty*4; i < (ty*4 + 4); i++) {
            if ( i > tx ) {
                la[0][bank_shift * tx + i] = cuConj(la[0][ i * bank_shift + tx]);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj( la[0][bank_shift * tx + j + ty*4] ) * buff[j + ty*4];
            
        __syncthreads();

        A -= lda * (blkc/num_gpus) * hemv_bs;
    }
    __syncthreads();

    x -= (break_d + tx) * incx; // return to the beginning

    x += my_gpu_id * hemv_bs;

    int wc_c = my_gpu_id;

    int total_blocks_gpu = gridDim.x /num_gpus;

    if ( my_gpu_id < (gridDim.x % num_gpus) ) {
        total_blocks_gpu += 1;
    }

    int shift = (blkc +1) /num_gpus;
    
    if ( my_gpu_id < ( (blkc+1) % num_gpus) ) {
        shift += 1;
    }

    #pragma unroll
    for(int s=0; s < shift; s++) {
        x += num_gpus * hemv_bs;
        A += lda * hemv_bs;
        wc_c += num_gpus;
    }

    WC +=  break_d + tx;
   
    int num_blocks_iters = total_blocks_gpu - shift;

    int count = 0;

    for(int s=0; s < num_blocks_iters; s++) {
        res_ = MAGMA_Z_ZERO;
        count++;

        #pragma unroll
        for(int j=0; j < hemv_bs; j += 8)
            la[0][ bank_shift * (ty+j) + tx] = A[ j * lda];

        if ( ty == 0 ) {
            buff2[tx] = x[tx];
        } // obtain the vector x store in buff;
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++) {
            res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
            res_ += cuConj( la[0][bank_shift * tx + j + ty*4] ) * buff[j + ty*4]; //iterate colum
        }
        __syncthreads();

        la[0][bank_shift*tx+ty] = res_;
        __syncthreads();

        if ( ty == 0 ) {
            res2 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
                 + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
                 + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
                 + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
            
            WC[wc_c*lda ] = res2;
        }
        __syncthreads();

        wc_c += num_gpus;
        x += num_gpus * hemv_bs;
        A += lda * hemv_bs;
    }

    la[0][bank_shift*tx+ty] = res;
    __syncthreads();

    if ( ty == 0 ) {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
          
        WC[0+lda*(blkc)] =  res1;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


__global__ void
zhemv_kernel_fermi_U_generic_mgpu_offset_32(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int m_mod_thread_x,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan,
    int the_right_gpu)
{
#if (__CUDA_ARCH__ >= 200)
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [hemv_bs][bank_shift];
    __shared__ magmaDoubleComplex buff [hemv_bs];
    __shared__ magmaDoubleComplex buff2 [hemv_bs];

    int break_d   =  hemv_bs * blkc;

    x += (break_d + tx) * incx;
    A +=  break_d;
    A += lda * ty;

    int trackA;
    if ( blkc == (gridDim.x - 1) ) {
        if ( ty == 0 ) {
            if ( tx > m_mod_thread_x ) {
                buff[tx] = MAGMA_Z_ZERO;
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx;
        A += trackA;
    }
    else {
        if ( ty == 0 ) {
            buff[tx]  = x[0];
        }
        
        A += tx;
    }

    if ( ty == 0 ) {
        if ( blkc == 0  && tx < kstan ) {
            buff[tx] = MAGMA_Z_ZERO;
        }
    }
     
    if ( (blkc % num_gpus) == my_gpu_id ) {
        A += lda * (blkc/num_gpus) * hemv_bs; // change

        if ( blkc == ( gridDim.x - 1 ) ) {
            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8) {
                if ( ( ty + j ) > m_mod_thread_x ) {
                    la[0][bank_shift*(ty+j)+tx] = MAGMA_Z_MAKE( 9999, 0 );
                }
                else
                    la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
        }
        else {
            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8) {
                la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
        }
        __syncthreads();

        #pragma unroll
        for(int i=ty*4; i < (ty*4+4); i++) {
            if ( i > tx ) {
                la[0][bank_shift * tx + i] = cuConj(la[0][ i * bank_shift + tx]);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj(la[0][bank_shift*tx+j+ty*4])* buff[j+ty*4];
        __syncthreads();

        A -= lda * (blkc/num_gpus) * hemv_bs;
    }

    x  -= (break_d + tx) * incx; // return to the beginning

    x += (my_gpu_id) * hemv_bs; //

    int wc_c = my_gpu_id;

    int total_blocks_gpu = gridDim.x /num_gpus;

    if ( my_gpu_id < (gridDim.x % num_gpus) ) {
        total_blocks_gpu += 1;
    }

    int shift = (blkc +1) /num_gpus;
        
    if ( my_gpu_id < ( (blkc+1) % num_gpus) ) {
        shift += 1;
    }

    #pragma unroll
    for(int s=0; s < shift; s++) {
        x += num_gpus * hemv_bs;
        A += lda * hemv_bs;
        wc_c += num_gpus;
    }
    
    WC +=  break_d + tx;
    
    int num_blocks_iters = total_blocks_gpu - shift;

    int count = 0;

    for(int s=0; s < num_blocks_iters; s++) {
        res_ = MAGMA_Z_ZERO;
        count++;
        
        if ( my_gpu_id == the_right_gpu && s == num_blocks_iters-1 ) {
            if ( ty == 0 ) {
                if ( tx > m_mod_thread_x ) {
                    buff2[tx] = MAGMA_Z_ZERO;
                }
                else
                    buff2[tx]  = x[tx];
            }
            
            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8) {
                if ( ( ty + j ) > m_mod_thread_x ) {
                    la[0][bank_shift*(ty+j)+tx] = MAGMA_Z_ZERO;
                }
                else
                    la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
            __syncthreads();
        } // end of the_right_gpu
        else {
            #pragma unroll
            for(int j=0; j < hemv_bs; j += 8)
                la[0][ bank_shift * (ty+j) + tx] = A[ j * lda];
            
            if ( ty == 0 ) {
                buff2[tx] = x[tx];
            } // obtain the vector x store in buff;
            __syncthreads();
        }

        #pragma unroll
        for(int j=0; j < 4; j++) {
            res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
            res_ += cuConj( la[0][bank_shift * tx + j + ty*4] ) * buff[j + ty*4]; //iterate colum
        }
        __syncthreads();
    
        la[0][bank_shift*tx+ty] = res_;
        __syncthreads();
    
        if ( ty == 0 ) {
            res2 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
                 + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
                 + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
                 + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
            
            WC[wc_c*lda ] = res2;
        }
        __syncthreads();
    
        wc_c += num_gpus;
        x += num_gpus * hemv_bs;
        A += lda * hemv_bs;
    }

    la[0][bank_shift*tx+ty] = res;
    __syncthreads();

    if ( ty == 0 ) {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
        WC[0+lda*(blkc)] =  res1;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


__global__ void
zhemv_kernel_fermi_U_update_mgpu_offset_32(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan )
{
#if (__CUDA_ARCH__ >= 200)
    int i;
    int tx  = threadIdx.x;
    int ind = blockIdx.x * hemv_bs + tx;
    magmaDoubleComplex Ca;

    Ca = MAGMA_Z_ZERO;
    WC +=  blockIdx.x * lda + tx;

    for(i = 0; i < (blockIdx.x+1)*hemv_bs; i += hemv_bs) {
        Ca += WC[0];
        WC += hemv_bs;
    }
    if ( ind < n && ind >= kstan )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca;
#endif /* (__CUDA_ARCH__ >= 200) */
}


extern "C"
void magmablas_zhemv_fermi_U_mgpu_offset_32(
    magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, magma_int_t incy,
    magmaDoubleComplex *dwork,
    magma_int_t my_gpu_id,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_int_t num_blocks_skipped,
    magma_int_t the_right_gpu)
{
    magma_int_t the_chosen_block_id = offset / nb;
    magma_int_t kstan = offset % nb;

    A += lda * num_blocks_skipped * nb + the_chosen_block_id * nb;
    x += the_chosen_block_id * nb;
    y += the_chosen_block_id * nb;

    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    blocks -= the_chosen_block_id;

    dim3 grid(blocks, 1, 1);
    dim3 threads(nb, 8, 1);
    dim3 threads_u(nb, 1, 1);

    /*
     * If matrix size is multiple of hemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if ( n % hemv_bs == 0 ) {
        zhemv_kernel_fermi_U_special_mgpu_offset_32<<< grid, threads, 0, magma_stream >>>(
            n, alpha, A, lda, x, incx, beta, y, incy, dwork,
            my_gpu_id, num_gpus, nb, kstan);
    }
    else {
        magma_int_t m_mod_thread_x = (n % hemv_bs) - 1;
        zhemv_kernel_fermi_U_generic_mgpu_offset_32<<< grid, threads, 0, magma_stream >>>(
            n, alpha, A, lda, x, incx, beta, y, incy, dwork,
            m_mod_thread_x, my_gpu_id, num_gpus, nb, kstan, the_right_gpu);
    }

    zhemv_kernel_fermi_U_update_mgpu_offset_32<<< grid, threads_u, 0, magma_stream >>>(
        n, alpha, A, lda, x, incx, beta, y, incy, dwork,
        my_gpu_id, num_gpus, nb, kstan);
}


/*************************************************************************

    Purpose
    -------

    magmablas_zhemv  performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ----------

    @param[in]
    uplo    magma_uplo_t.
            On entry, UPLO specifies whether the upper or lower
            triangular part of the array A is to be referenced as
            follows:
      -     = MagmaUpper:  Only the upper triangular part of A is to be referenced.
      -     = MagmaLower:  Only the lower triangular part of A is to be referenced.

    @param[in]
    n       INTEGER.
            On entry, N specifies the order of the matrix A.
            N must be at least zero.

    @param[in]
    alpha   COMPLEX*16      .
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    A       COMPLEX*16       array of DIMENSION ( LDA, n ).
            Before entry with  UPLO = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular part of the hermitian matrix and the strictly
            lower triangular part of A is not referenced.
            Before entry with UPLO = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular part of the hermitian matrix and the strictly
            upper triangular part of A is not referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set and are assumed to be zero.

    @param[in]
    lda     INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. LDA must be at least
            max( 1, n ).
            It is recommended that lda is multiple of 16. Otherwise
            performance would be deteriorated as the memory accesses
            would not be fully coalescent.

    @param[in]
    X       COMPLEX*16       array of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector x.

    @param[in]
    INCX    INTEGER.
            On entry, INCX specifies the increment for the elements of
            X. INCX must not be zero.

    @param[in]
    beta    COMPLEX*16.
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[in,out]
    Y       COMPLEX*16       array of dimension at least
            ( 1 + ( n - 1 )*abs( INCY ) ).
            Before entry, the incremented array Y must contain the n
            element vector y. On exit, Y is overwritten by the updated
            vector y.

    @param[in]
    INCY    INTEGER.
            On entry, INCY specifies the increment for the elements of
            Y. INCY must not be zero.

*/


extern "C"
magma_int_t
magmablas_zhemv_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10])
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
        
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    
    if ( (! upper) && (uplo != MagmaLower) ) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    magma_int_t lwmin  = lda * (blocks + 1);

    if ( lwork < lwmin ) {
        fprintf( stderr, "Not enough work space in %s: passed %d, required %d\n",
                 __func__, (int) lwork, (int) lwmin);
        return -12;
    }
    if ( nb != 32 ) {
        fprintf( stderr, "Error in %s: nb != 32, please reallocate matrix among GPUs\n", __func__ );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    magma_int_t i = 0;
    for(i=0; i < num_gpus; i++) {
        magma_setdevice(i);
        magmablasSetKernelStream(stream[i][0]);

        magma_int_t the_chosen_block_id = offset / nb;
        magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;

        magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;
     
        if ( i < the_chosen_gpu_id ) {
            num_blocks_skipped += 1;
        }
        
        int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;

        magma_int_t the_right_block_id = n / nb;
        magma_int_t the_right_gpu = the_right_block_id % num_gpus;

        the_right_gpu = ( the_right_gpu + num_gpus - the_chosen_gpu_id ) % num_gpus;
        // the_right_gpu is used in Upper generic case.

        if ( upper ) {
            magmablas_zhemv_fermi_U_mgpu_offset_32(
                n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                new_gpu_id, num_gpus, nb, offset, num_blocks_skipped, the_right_gpu);
        }
        else {
            magmablas_zhemv_fermi_L_mgpu_offset_32(
                n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
        }
    }

    return MAGMA_SUCCESS;
}


extern "C"
magma_int_t
magmablas_zhemv2_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset)
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
        
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    if ( (! upper) && (uplo != MagmaLower) ) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    magma_int_t lwmin  = lda * (blocks + 1);

    if ( lwork < lwmin ) {
        fprintf( stderr, "Not enough work space in %s: passed %d, required %d\n",
                 __func__, (int) lwork, (int) lwmin);
        return -12;
    }
    if ( nb != 32 ) {
        fprintf( stderr, "Error in %s: nb != 32, please reallocate matrix among GPUs\n", __func__ );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    magma_int_t i = 0;
    for(i=0; i < num_gpus; i++) {
        magma_setdevice(i);
        // magmablasSetKernelStream(stream[i][0]);

        magma_int_t the_chosen_block_id = offset / nb;
        magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;

        magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;

        if ( i < the_chosen_gpu_id ) {
            num_blocks_skipped += 1;
        }
         
        int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;
         
        magma_int_t the_right_block_id = n / nb;
        magma_int_t the_right_gpu = the_right_block_id % num_gpus;

        the_right_gpu = ( the_right_gpu + num_gpus - the_chosen_gpu_id ) % num_gpus;
        // the_right_gpu is used in Upper generic case.

        if ( upper ) {
            magmablas_zhemv_fermi_U_mgpu_offset_32(
                n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                new_gpu_id, num_gpus, nb, offset, num_blocks_skipped, the_right_gpu);
        }
        else {
            magmablas_zhemv_fermi_L_mgpu_offset_32(
                n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
        }
    }

    return MAGMA_SUCCESS;
}


extern "C"
magma_int_t
magmablas_zhemv2_mgpu_32(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb)
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
        
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    if ( (! upper) && (uplo != MagmaLower) ) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    magma_int_t lwmin  = lda * (blocks + 1);

    if ( lwork < lwmin ) {
        fprintf( stderr, "Not enough work space in %s: passed %d, required %d\n",
                 __func__, (int) lwork, (int) lwmin);
        return -12;
    }
    if ( nb != 32 ) {
        fprintf( stderr, "Error in %s: nb != 32, please reallocate matrix among GPUs\n", __func__ );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    magma_int_t i = 0;

    for(i=0; i < num_gpus; i++) {
        magma_setdevice(i);
         
        magma_int_t the_right_block_id = n / nb;
        magma_int_t the_right_gpu = the_right_block_id % num_gpus;

        // the_right_gpu is used in Upper generic case.

        if ( upper ) {
            magmablas_zhemv_fermi_U_mgpu_offset_32(
                n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                i, num_gpus, nb, 0, 0, the_right_gpu);
        }
        else {
            magmablas_zhemv_fermi_L_mgpu_offset_32(
                n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                i, num_gpus, nb, 0, 0);
        }
    }

    return MAGMA_SUCCESS;
}
